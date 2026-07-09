from __future__ import annotations

import re
from pathlib import Path


class BoundaryGuard:
    sensitive_file_patterns = (".env", "secret", "key", "token", "credential")
    injection_patterns = (
        "ignore previous instructions",
        "忽略以上指令",
        "system prompt",
        "developer message",
        "tool call",
        "删除文件",
        "读取密钥",
    )
    shell_allowed_prefixes = ("pwd", "ls", "cat", "head", "wc", "python -m pytest")
    shell_forbidden_tokens = (
        "rm",
        "mv",
        "curl",
        "wget",
        "ssh",
        "sudo",
        "chmod",
        "chown",
        "env",
        "&&",
        "|",
        ";",
        "`",
        "$(",
    )

    def ensure_workspace_path(self, path: str, workspace: Path) -> Path:
        candidate = (workspace / path).resolve()
        workspace_resolved = workspace.resolve()
        if workspace_resolved not in candidate.parents and candidate != workspace_resolved:
            raise ValueError(f"path outside workspace: {path}")
        lowered = candidate.name.lower()
        if any(pattern in lowered for pattern in self.sensitive_file_patterns):
            raise ValueError(f"sensitive file access blocked: {path}")
        return candidate

    def ensure_output_path(self, path: str, workspace: Path) -> Path:
        candidate = self.ensure_workspace_path(path, workspace)
        outputs_dir = (workspace / "outputs").resolve()
        if outputs_dir not in candidate.parents and candidate != outputs_dir:
            raise ValueError(f"write outside outputs blocked: {path}")
        return candidate

    def scan_untrusted_text(self, text: str) -> list[str]:
        lowered = text.lower()
        flags = []
        if any(pattern in lowered for pattern in self.injection_patterns):
            flags.append("prompt_injection_detected")
        if re.search(r"\b\d{17}[\dXx]\b", text):
            flags.append("privacy_identifier_detected")
        if re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text):
            flags.append("privacy_phone_detected")
        return flags

    def mask_privacy(self, text: str) -> str:
        text = re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email masked]", text)
        return re.sub(r"\b(\d{6})\d{8}(\d{3}[\dXx])\b", r"\1********\2", text)

    def validate_shell_command(self, command: str) -> tuple[bool, str]:
        stripped = command.strip()
        if any(token in stripped for token in self.shell_forbidden_tokens):
            return False, "shell_blocked"
        if not any(
            stripped == prefix or stripped.startswith(prefix + " ")
            for prefix in self.shell_allowed_prefixes
        ):
            return False, "shell_not_allowlisted"
        return True, "allowed"
