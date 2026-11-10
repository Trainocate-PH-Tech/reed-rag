# RAG System Prototype

## Commands to Run

### Running Tests

For all tests:

```
pytest
```

For a specific test:

```
pytest tests/unit/test_operations.py
```

### Command Line

To display chunks:

```
rag process text-file ./samples/test.txt --chunk_size=10 --stride=5
```
