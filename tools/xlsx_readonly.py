"""Read cached cell values from OOXML without modifying the source workbook."""
from pathlib import Path
import zipfile, xml.etree.ElementTree as ET, posixpath, re
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
def col_num(addr):
 n=0
 for ch in re.match('[A-Z]+',addr).group():n=n*26+ord(ch)-64
 return n
class CachedWorkbook:
 def __init__(self,path):
  self.path=Path(path);self._cache={}
  with zipfile.ZipFile(self.path) as z:
   self.strings=[]
   if 'xl/sharedStrings.xml' in z.namelist():
    for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS):
     self.strings.append(''.join(t.text or '' for t in si.findall('.//s:t',NS)))
   rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
   self.paths={}
   for s in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
    target=rels[s.attrib['{'+REL+'}id']]
    self.paths[s.attrib['name']]=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
 def rows(self,sheet):
  if sheet in self._cache:return self._cache[sheet]
  with zipfile.ZipFile(self.path) as z:root=ET.fromstring(z.read(self.paths[sheet]))
  rows=[]
  for row in root.findall('s:sheetData/s:row',NS):
   rid=int(row.attrib['r']); vals=[]
   while len(rows)<rid:rows.append([])
   for c in row.findall('s:c',NS):
    typ=c.get('t');v=c.find('s:v',NS);raw=v.text if v is not None else None
    if typ=='s':value=self.strings[int(raw)] if raw is not None else None
    elif typ=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//s:t',NS))
    elif typ=='e':raise ValueError(f'Error cell {sheet}!{c.attrib["r"]}: {raw}')
    elif typ=='b':value=raw=='1'
    elif typ=='str':value=raw or '' # OOXML string-formula cache may omit <v> for empty strings
    elif raw is None:
     if c.find('s:f',NS) is not None:raise ValueError(f'Formula without cached value: {sheet}!{c.attrib["r"]}')
     value=None
    elif typ=='str':value=raw
    else:
     try:value=float(raw);value=int(value) if value.is_integer() else value
     except ValueError:value=raw
    col=col_num(c.attrib['r'])
    while len(vals)<col:vals.append(None)
    vals[col-1]=value
   rows[rid-1]=vals
  self._cache[sheet]=rows;return rows
 def records(self,sheet,header_row):
  rows=self.rows(sheet);head=rows[header_row-1]
  if len(set(h for h in head if h is not None))!=len([h for h in head if h is not None]):raise ValueError(f'Duplicate headers: {sheet}')
  return [dict(zip(head, r+[None]*(len(head)-len(r))),_excel_row=i) for i,r in enumerate(rows[header_row:],header_row+1) if any(v is not None for v in r)]
 def fields(self,sheet):
  rows=self.rows(sheet)
  return {r[0]:r[1] for r in rows[3:] if len(r)>1 and r[0] is not None}
