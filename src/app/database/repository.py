from app.database.documents import from_record, to_metadata

class DocumentRepository:
    def __init__(self, collection):
        self.collection = collection

    def create(self, document):
        self.collection.add(
            ids=[document.id], documents=[document.content],
            embeddings=[document.embedding], metadatas=[to_metadata(document)],
        )
        return document

    def _records(self, result):
        return [from_record(record_id, result["documents"][i],
                            result["embeddings"][i], result["metadatas"][i])
                for i, record_id in enumerate(result["ids"])]

    def get(self, record_id):
        result = self.collection.get(
            ids=[record_id], include=["documents", "embeddings", "metadatas"],
        )
        records = self._records(result)
        return records[0] if records else None

    def delete(self, record_id):
        if self.get(record_id) is None:
            return False
        self.collection.delete(ids=[record_id])
        return True

    def list(self, limit=100, offset=0):
        if (type(limit) is not int or type(offset) is not int
                or limit < 1 or offset < 0):
            raise ValueError("Invalid pagination")
        return self._records(self.collection.get(
            limit=limit, offset=offset,
            include=["documents", "embeddings", "metadatas"],
        ))
