# Third-party notices

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
