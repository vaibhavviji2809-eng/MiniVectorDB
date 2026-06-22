# Storage Engine

MiniVectorDB stores data in the same order described in the roadmap:

`Collection -> Segments -> Pages -> Vectors`

Each vector record contains:

- `id`
- `vector`
- `metadata`
- optional `text`

