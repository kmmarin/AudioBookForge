-- Flattens a document into speakable plain text.
-- Headings become marker paragraphs so Python can split chapters;
-- things that read badly aloud (images, notes, tables, code) are dropped.
function Header(el)
  local inlines = pandoc.List({pandoc.Str("@@H" .. el.level .. "@@"), pandoc.Space()})
  inlines:extend(el.content)
  return pandoc.Para(inlines)
end
function HorizontalRule(el) return pandoc.Para({pandoc.Str("@@BREAK@@")}) end
function Image(el) return {} end
function Note(el) return {} end
function Link(el) return el.content end
function Table(el) return {} end
function CodeBlock(el) return {} end
function RawBlock(el) return {} end
function RawInline(el) return {} end
