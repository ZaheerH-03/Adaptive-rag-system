from dataclasses import dataclass
from typing import Optional, Dict

@dataclass()
class DocUnit:
    text: str
    filename: str
    file_type: str
    page_num: Optional[int] = None
    slide_num: Optional[int] = None
    section_title: Optional[int] = None
    extra_meta : Optional[Dict] = None

    def to_metadata(self) -> Dict:
        meta = {
            "filename" : self.filename,
            "file_type": self.file_type,
        }
        if self.page_num is not None:
            meta['page_num'] = self.page_num
        if self.slide_num is not None:
            meta['page_num'] = self.slide_num
        if self.section_title is not None:
            meta['section_title'] = self.section_title
        if self.extra_meta is not None:
            meta.update(self.extra_meta)

        return meta