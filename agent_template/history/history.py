from dataclasses import dataclass


@dataclass(frozen=False, slots=True, kw_only=True)
class History:
    content: list[dict[str, str]]

    def add(self, role: str, content: str) -> None:
        self.content.append({"role": role, "content": content})
