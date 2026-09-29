"""
Prompt generation and output parsing module for CTI (Cyber Threat Intelligence) analysis.

This module provides the `StepPrompt` class, which implements a multi-stage pipeline
for mapping cyber threat behaviors to the MITRE ATT&CK framework. The pipeline
consists of domain classification, behavior extraction, technique mapping,
and sub-technique granularization.

Dependencies:
    - template.py: Provides tactic dictionaries and technique lists per domain.
    - tech2id.json, tech2disc.json, tech2disc_concise.json, tech2tac.json,
      tech2subtech.json: ATT&CK knowledge base mapping files.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .template import all_enterprise, all_ICS, all_mobile, tactic_dict, tactic_dict_list

__all__ = ["StepPrompt"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output format templates (class-level constants for reuse)
# ---------------------------------------------------------------------------

_TACTIC_OUTPUT_FORMAT = """\
Please output strictly in the following format in English:
Tactic 1:
<tactic name>
1.
2.

Tactic 2:
<tactic name>
1.
2.
...
"""

_TECHNIQUE_OUTPUT_FORMAT = """\
Please output strictly in the following format in English: (technique name can only be the name of a technique, merging is not allowed; do not omit the output line starting with "Technique")
Technique 1: 
<technique name>
<Description of Behaviour>

Technique 2:
<technique name>
<Description of Behaviour>
...
"""

_IOC_PATTERN = re.compile(r'[!-~]{64,}|[a-zA-Z0-9]{32,}')

_TECH_ID_PATTERN = re.compile(r'T\d{4}\.\d{3}')
_TECH_ID_SHORT_PATTERN = re.compile(r'T\d{4}')

_SPECIAL_CHARS_BEHAVIOR = re.compile(r'[\[\]()*`]')
_SPECIAL_CHARS_TECHNIQUE = re.compile(r'[\[\]()*#`]')
_SPECIAL_CHARS_DESCRIBE = re.compile(r'[\[\]()*`]')

_PREFIX_TACTIC_NAME = "tactic name: "
_PREFIX_TACTIC_COLON = "Tactic: "
_PREFIX_TECHNIQUE_NAME = "technique name: "
_PREFIX_TECHNIQUE_COLON = "Technique: "
_PREFIX_DESCRIPTION = "Description of Behaviour: "
_PREFIX_DESCRIPTION_SHORT = "Description: "
_TECHNIQUE_MAPPING_HEADER = "### Technique Mapping"

_DOMAIN_ENTERPRISE = "enterprise"
_DOMAIN_MOBILE = "mobile"
_DOMAIN_ICS = "industrial control system"

_VALID_DOMAINS = {_DOMAIN_ENTERPRISE, _DOMAIN_MOBILE, _DOMAIN_ICS}


class StepPrompt:
    """Multi-stage CTI analysis pipeline for ATT&CK technique mapping.

    This class orchestrates a sequence of prompt-generation and output-parsing
    stages that together transform a raw threat intelligence report into a
    structured attack chain with ATT&CK technique annotations.

    Pipeline stages:
        0. Domain classification (Enterprise / Mobile / ICS)
        1. Behavior extraction grouped by tactic
        2. Technique mapping (with verification and enrichment)
        3. Sub-technique granularization

    Args:
        content: Raw CTI report text (IoC indicators will be stripped
            automatically).
        file_pathname: Path to the source CTI report file; used to derive
            the domain directory and report filename.
    """

    def __init__(self, content: str, file_pathname: str) -> None:
        # --- Knowledge base (loaded from JSON) ---
        self.tech2id_dict: Dict[str, Dict[str, str]] = self._load_json("tech2id.json")
        self.tech2disc_dict: Dict[str, Dict[str, str]] = self._load_json("tech2disc_concise.json")
        self.tech2disc_complete_dict: Dict[str, Dict[str, str]] = self._load_json("tech2disc.json")
        self.tech2tac_dict: Dict[str, Dict[str, List[str]]] = self._load_json("tech2tac.json")
        self.tech2subtech_dict: Dict[str, Dict[str, List[str]]] = self._load_json("tech2subtech.json")

        # --- Pipeline state ---
        self.valid_techniques_domain: Optional[Any] = None
        self.qwen3_recommend_subtech: Optional[Dict[str, List[str]]] = None
        self.domain: Optional[str] = None
        self.behaviors_json: Optional[Dict[str, List[str]]] = None

        # --- Report metadata ---
        self.content_origin: str = content
        self.content: str = self._delete_ioc(content)
        self.file_pathname: str = file_pathname
        self.domain_name: str = os.path.dirname(file_pathname)
        self.filename: str = Path(file_pathname).stem

        # --- Domain reference data (from template) ---
        self.tactic_dict: Dict[str, str] = tactic_dict
        self.tactic_dict_list: Dict[str, List[str]] = tactic_dict_list
        self.all_enterprise: str = all_enterprise
        self.all_mobile: str = all_mobile
        self.all_ICS: str = all_ICS
        self.technique_all_dict: Dict[str, str] = {
            _DOMAIN_ENTERPRISE: self.all_enterprise,
            _DOMAIN_MOBILE: self.all_mobile,
            _DOMAIN_ICS: self.all_ICS,
        }

        # --- Output format templates ---
        self.tactic_output_format: str = _TACTIC_OUTPUT_FORMAT
        self.technique_output_format: str = _TECHNIQUE_OUTPUT_FORMAT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(filename: str) -> Any:
        """Load a JSON file from the module's directory.

        Args:
            filename: Name of the JSON file (relative to this module).

        Returns:
            Parsed JSON content.
        """
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _delete_ioc(content: str) -> str:
        """Strip IoC-like strings (long hex/base64 tokens) from content.

        Replaces high-entropy strings (64+ printable ASCII chars or 32+
        alphanumeric chars) with ``[]`` to prevent hallucination interference.

        Args:
            content: Raw report text.

        Returns:
            Sanitized text with IoC patterns removed.
        """
        return _IOC_PATTERN.sub("[]", content)

    @staticmethod
    def _clean_text_behaviors(text: str) -> str:
        """Clean a behavior text line by removing punctuation and prefixes."""
        text = _SPECIAL_CHARS_BEHAVIOR.sub("", text)
        text = text.replace(_PREFIX_TACTIC_NAME, "").replace(_PREFIX_TACTIC_COLON, "")
        text = text.replace("<", "").replace(">", "")
        return text.strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """General-purpose text cleaner for technique lines."""
        text = _TECH_ID_PATTERN.sub("", text)
        text = _TECH_ID_SHORT_PATTERN.sub("", text)
        text = _SPECIAL_CHARS_TECHNIQUE.sub("", text)
        text = text.replace("<", "").replace(">", "").replace(_TECHNIQUE_MAPPING_HEADER, "")
        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _parse_binary_result(output: str, key_a: str, key_b: str) -> Tuple[str, str]:
        """Parse LLM output for a binary classification result.

        Args:
            output: Raw LLM output text.
            key_a: Label prefix for the score line (e.g., ``"Score"``).
            key_b: Label prefix for the rationale line (e.g., ``"Rationale"``).

        Returns:
            A tuple of ``(score, rationale)``.
        """
        lines = output.split("\n")
        rationale = ""
        score = ""
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith(key_a.lower()):
                score = stripped[len(key_a) + 1 :]
            elif lower.startswith(key_b.lower()):
                rationale = stripped[len(key_b) + 1 :]
            elif lower.startswith(f"**{key_a}**".lower()):
                score = stripped[len(key_a) + 5 :]
            elif lower.startswith(f"**{key_b}**".lower()):
                rationale = stripped[len(key_b) + 5 :]
            else:
                rationale = rationale + "\n" + stripped
        return score, rationale

    # ------------------------------------------------------------------
    # Stage 0: Domain classification
    # ------------------------------------------------------------------

    def get_0_domain_prompt(self) -> str:
        """Generate prompt for ATT&CK domain classification.

        Returns:
            Prompt string asking the LLM to classify the report into
            Enterprise, Mobile, or Industrial Control System.
        """
        return f"""{self.content}
上述内容的domain属于Enterprise、Mobile还是Industrial Control System?请仅输出Enterprise或Mobile或Industrial Control System。"""

    def get_domain(self, output: str) -> str:
        """Parse the domain classification result and initialize domain state.

        Args:
            output: Raw LLM response containing the domain name.

        Returns:
            Normalized domain string (``"enterprise"``, ``"mobile"``, or
            ``"industrial control system"``), or an empty string on error.
        """
        output_clean = re.sub(r"[\[\]()*]", "", output).lower()
        if output_clean in _VALID_DOMAINS:
            self.domain = output_clean
            self.behaviors_json = {tactic: [] for tactic in self.tactic_dict_list[self.domain]}
            self.valid_techniques_domain = self.tech2disc_dict[self.domain].keys()
            return self.domain
        else:
            logger.error("Domain classification failed; received output: %s", output)
            return ""

    # ------------------------------------------------------------------
    # Stage 1: Behavior extraction
    # ------------------------------------------------------------------

    def get_1_behaviors_prompt(self) -> Optional[str]:
        """Generate prompt for extracting attack behaviors grouped by tactic.

        Returns:
            Prompt string, or ``None`` if the domain has not been set.
        """
        if self.domain is None:
            logger.error("Cannot build behavior prompt: domain not set.")
            return None

        return f"""{self.content}
根据上述内容，尽量详尽地提取每个网络行为，使用不同资产IoC的相同行为应总结为单个行为，每个行为用一句话描述，按照战术{self.tactic_dict[self.domain]}进行归类。没有相关攻击行为的战术不需要输出。

{self.tactic_output_format}"""

    def parse_behaviors_to_json(self, output: str) -> None:
        """Parse behavior extraction output into ``self.behaviors_json``.

        Args:
            output: Raw LLM response with tactic-grouped behaviors.
        """
        if self.behaviors_json is None:
            logger.warning("behaviors_json is None; skipping parse.")
            return

        lines = output.split("\n")
        i = 0
        while i < len(lines):
            line_content = lines[i].strip().replace("#", "").replace("*", "")
            lower = line_content.lower()

            if lower.startswith("tactic"):
                if ":" in line_content:
                    parts = line_content.split(":", 1)
                    tactic_name = parts[1].strip() if len(parts) > 1 else ""
                    offset = 1
                else:
                    tactic_name = ""
                    offset = 1

                if not tactic_name or tactic_name not in self.tactic_dict_list[self.domain]:
                    if i + 1 >= len(lines):
                        break
                    tactic_name = self._clean_text_behaviors(lines[i + 1].strip())
                    if tactic_name not in self.tactic_dict_list[self.domain]:
                        logger.error("Invalid tactic: %s", tactic_name)
                        break
                    offset = 2

                while i + offset < len(lines) and lines[i + offset][:1].isdigit():
                    self.behaviors_json[tactic_name].append(lines[i + offset])
                    offset += 1
                i += offset

            elif (
                self._clean_text_behaviors(line_content).split(":", 1)[0].strip()
                in self.tactic_dict_list[self.domain]
            ):
                tactic_name = self._clean_text_behaviors(line_content).split(":", 1)[0].strip()
                offset = 1
                while i + offset < len(lines) and lines[i + offset][:1].isdigit():
                    self.behaviors_json[tactic_name].append(lines[i + offset])
                    offset += 1
                i += offset
            else:
                i += 1

    def get_1_double_behaviors_prompt(self) -> str:
        """Generate a follow-up prompt to discover any residual behaviors.

        Returns:
            Prompt string that asks the LLM to supplement the existing
            behavior list with any missed or logically inferable behaviors.
        """
        json_str = str(self.behaviors_json)
        return f"""{self.content}
根据上述内容，尽量详尽地提取每个攻击行为，使用不同资产IoC的相同行为应总结为单个行为，在以下结果的基础上：
{json_str}
请进行补充，寻找任何可能残留的、可逻辑推导的攻击行为迹象，仅输出新增的攻击行为，每个行为用一句话描述。没有相关攻击行为的战术不需要输出。
{self.tactic_output_format}"""

    # ------------------------------------------------------------------
    # Stage 2: Technique mapping
    # ------------------------------------------------------------------

    def get_techniques_intersection_diff(
        self,
        document_level_result: Dict[str, Any],
        output_qwen3_dict: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Compute intersection and symmetric difference of two technique maps.

        Args:
            document_level_result: First technique dict.
            output_qwen3_dict: Second technique dict.

        Returns:
            A tuple of ``(intersection_result, diff_result)``.
        """
        intersection_result: Dict[str, Any] = {}
        diff_result: List[str] = []

        for tech in output_qwen3_dict:
            if tech in document_level_result:
                intersection_result[tech] = output_qwen3_dict[tech]
            else:
                diff_result.append(tech)

        for tech in document_level_result:
            if tech not in output_qwen3_dict:
                diff_result.append(tech)

        return intersection_result, diff_result

    def get_2b_diff_behav(self, tech: str) -> Optional[str]:
        """Generate prompt to verify whether a technique exists in the report.

        Args:
            tech: The technique name to verify.

        Returns:
            Prompt string, or ``None`` if domain is not set.
        """
        if self.domain is None:
            logger.error("Cannot build diff-behav prompt: domain not set.")
            return None

        official_desc = self.tech2disc_complete_dict[self.domain].get(tech, "")
        return f"""基于原文：{self.content}
判断提取的技术“{tech}”是否在原文中存在。
该技术的官方描述为“{official_desc}”
若存在，则基于威胁情报原文为该技术补充合适的行为描述（使用英文，且不需要额外的原因或说明）；若不存在，则给出可能造成误判的相关行为描述，并给出不符合该技术的原因。

输出格式为：
该技术是否在原文中存在: 是/否
与该技术相关的行为描述: xxx"""

    def get_2c_enrich(self, intersection_result: Dict[str, Any]) -> Optional[str]:
        """Generate prompt to discover additional techniques beyond those extracted.

        Args:
            intersection_result: Currently identified technique mappings.

        Returns:
            Prompt string, or ``None`` if domain is not set.
        """
        if self.domain is None:
            logger.error("Cannot build enrich prompt: domain not set.")
            return None

        return f"""基于原文：{self.content}
和已经提取的技术：
{str(intersection_result)}
请进行补充，寻找任何可能残留的、可逻辑推导的技术，仅输出新增的技术，每个技术的行为用一句话描述。
{self.technique_all_dict[self.domain]}

{self.technique_output_format}"""

    # ------------------------------------------------------------------
    # Stage 2 helpers (parsing)
    # ------------------------------------------------------------------

    def parse_techniques_to_json(self, output: str) -> Dict[str, Dict[str, str]]:
        """Parse technique listing output into a structured dict.

        Args:
            output: Raw LLM response listing techniques and descriptions.

        Returns:
            A dict of ``{filename: {technique_name: description}}``.
        """
        techniques: Dict[str, str] = {}
        lines = output.split("\n")

        # Preprocess: remove special characters and filter empty lines
        cleaned_lines = [self._clean_text(line) for line in lines if self._clean_text(line)]

        i = 0
        while i < len(cleaned_lines):
            line = cleaned_lines[i]

            if line.lower().startswith("technique"):
                if ":" in line:
                    parts = line.split(":", 1)
                    tech_name = parts[1].strip() if len(parts) > 1 else ""
                else:
                    tech_name = ""

                if not tech_name and i + 1 < len(cleaned_lines):
                    tech_name = cleaned_lines[i + 1]
                    i += 1

                description = ""
                if i + 1 < len(cleaned_lines):
                    description = cleaned_lines[i + 1]
                    i += 1

                if tech_name:
                    techniques[tech_name] = description

                i += 1
            else:
                i += 1

        return {self.filename: techniques}

    # ------------------------------------------------------------------
    # Stage 3: Sub-technique granularization
    # ------------------------------------------------------------------

    def get_subtech_prompt(self, techname: str, techdisc: str) -> str:
        """Generate prompt for selecting the best-matching sub-technique.

        Args:
            techname: Parent technique name.
            techdisc: Description of the observed behavior.

        Returns:
            Prompt string asking the LLM to pick the most appropriate
            sub-technique from the candidates.
        """
        subtech_list = self.tech2subtech_dict[self.domain][techname]
        subtech_str = " ".join(f'"{item}"' for item in subtech_list)
        subtech_output = "或".join(subtech_list)

        ttp_disc_list = []
        for item in subtech_list:
            if item in self.tech2disc_dict[self.domain]:
                ttp_disc_list.append(f"{item}: {self.tech2disc_dict[self.domain][item]}")
        ttp_disc_str = "\n".join(ttp_disc_list)

        return f"""{techdisc}
判断上述描述最符合以下哪个名称：
{subtech_str}
这些技术的官方描述如下，可供参考：
{ttp_disc_str}
若所有子技术均不完全匹配，优先输出{subtech_list[0]}
请仅输出{subtech_output}"""

    def get_2b_diff_behav_subtech(self, tech: str, behav: str) -> Optional[str]:
        """Generate prompt to verify a sub-technique against a behavior description.

        Args:
            tech: Sub-technique name to verify.
            behav: Behavior description to check against.

        Returns:
            Prompt string, or ``None`` if domain is not set.
        """
        if self.domain is None:
            logger.error("Cannot build subtech diff-behav prompt: domain not set.")
            return None

        official_desc = self.tech2disc_complete_dict[self.domain].get(tech, "")
        return f"""基于描述：{behav}
判断提取的技术“{tech}”是否在描述中存在。
该技术的官方描述为“{official_desc}”
若存在，则基于威胁情报原文为该技术补充合适的行为描述（使用英文，且不需要额外的原因或说明）；若不存在，则给出可能造成误判的相关行为描述，并给出不符合该技术的原因。

输出格式为：
该技术是否在原文中存在: 是/否
与该技术相关的行为描述: xxx"""


    # ------------------------------------------------------------------
    # Qwen3-specific prompt helpers
    # ------------------------------------------------------------------

    def get_qwen3_document_prompt(
        self, behaviors_str: str, subtech: bool = False
    ) -> str:
        """Build a Qwen3-format document-level behavior-to-technique prompt.

        Args:
            behaviors_str: Behavior descriptions to map.
            subtech: If ``True``, require precision at the sub-technique level.

        Returns:
            Formatted Qwen3 chat prompt string.
        """
        subtech_clause = "要求精确到子技术级别。" if subtech else ""
        return (
            "<|im_start|>user\n"
            f"{behaviors_str}\n"
            f"    将以上行为映射到ATT&CK的技术，每行输出一个技术名。{subtech_clause}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def get_qwen3_sentence_prompt(
        self, behav: str, subtech: bool = False
    ) -> str:
        """Build a Qwen3-format sentence-level behavior-to-technique prompt.

        Args:
            behav: Single behavior description to map.
            subtech: If ``True``, require precision at the sub-technique level.

        Returns:
            Formatted Qwen3 chat prompt string.
        """
        subtech_clause = "要求精确到子技术级别。" if subtech else ""
        return (
            "<|im_start|>user\n"
            f"{behav}\n"
            f"将以上行为映射到ATT&CK的技术。{subtech_clause}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
