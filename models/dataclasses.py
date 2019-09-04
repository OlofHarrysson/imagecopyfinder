from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class Entry:
  im_type: str
  match_id: int
  im_id: int