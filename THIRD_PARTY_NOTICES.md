# Third-party notices

## SeeStory

Watch and read's scene choice and prompts (`app/scenes.py`), the drawing settings in
`clone/server.py` (`/api/draw`) and the reader's slow movement on each picture are
adapted from [SeeStory](https://github.com/pgotta/SeeStory).

```
MIT License

Copyright (c) 2026 pgotta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Project Gutenberg

**Find a free book** (`app/gutenberg.py`) searches Project Gutenberg's catalog
(https://www.gutenberg.org/ebooks/offline_catalogs.html), which Project Gutenberg
has placed in the public domain, and fetches books from Project Gutenberg when a
user adds one. It follows Project Gutenberg's terms of use
(https://www.gutenberg.org/policy/terms_of_use.html):
- it downloads the catalog weekly instead of crawling the site;
- it fetches one book per user action, at most 50 a day, with a user-agent naming this project;
- it fetches only books whose record says "Public domain in the USA."

Each book keeps the Project Gutenberg License it comes with
(https://www.gutenberg.org/policy/license.html). Audiobook Forge is not
affiliated with Project Gutenberg; "Project Gutenberg" is a registered trademark
of the Project Gutenberg Literary Archive Foundation.
