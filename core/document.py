from dataclasses import dataclass, field
from typing import Optional, Dict
import uuid


@dataclass()
class DocUnit:
    text: str
    filename: str
    file_type: str
    page_num: Optional[int] = None
    slide_num: Optional[int] = None
    section_title: Optional[str] = None
    extra_meta : Optional[Dict] = None

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> Dict:
        meta = {
            "filename" : self.filename,
            "file_type": self.file_type,
        }
        if self.page_num is not None:
            meta['page_num'] = self.page_num
        if self.slide_num is not None:
            meta['slide_num'] = self.slide_num
        if self.section_title is not None:
            meta['section_title'] = self.section_title
        if self.extra_meta is not None:
            meta.update(self.extra_meta)

        return meta

@dataclass
class Chunk:
    text: str
    metadata: Dict
    parent_id: str

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))