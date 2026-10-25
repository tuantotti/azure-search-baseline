"""Validated SDK options and a separate unrestricted Azure REST body."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawSearchRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {"search": "nghỉ phép", "top": 10, "filter": "is_public eq true"}
        },
    )
    search: str | None = None
    filter: str | None = None
    top: int | None = Field(default=None, ge=1, le=1000)
    select: str | None = None
    count: bool | None = None
    vectorQueries: list[dict[str, Any]] | None = None
    queryType: str | None = None
    semanticConfiguration: str | None = None


class VectorQuery(StrictModel):
    vector: list[float] | None = None
    text: str | None = None
    fields: str = "content_vector"
    k_nearest_neighbors: int = Field(default=50, ge=1, le=1000)
    exhaustive: bool = False
    weight: float = Field(default=1, gt=0)
    oversampling: float | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_input(self) -> "VectorQuery":
        if (self.vector is None) == (self.text is None):
            raise ValueError("Provide exactly one of vector or text")
        if self.vector == [] or self.text == "":
            raise ValueError("Vector/text must not be empty")
        return self


class SearchOptions(StrictModel):
    """Named arguments supported by SearchClient.search in the pinned SDK."""

    search_text: str | None = None
    include_total_count: bool | None = None
    facets: list[str] | None = None
    filter: str | None = None
    highlight_fields: str | None = None
    highlight_pre_tag: str | None = None
    highlight_post_tag: str | None = None
    minimum_coverage: float | None = Field(default=None, ge=0, le=100)
    order_by: list[str] | None = None
    query_type: str | None = None
    scoring_parameters: list[str] | None = None
    scoring_profile: str | None = None
    semantic_query: str | None = None
    search_fields: list[str] | None = None
    search_mode: str | None = None
    query_answer: str | None = None
    query_answer_count: int | None = None
    query_answer_threshold: float | None = None
    query_caption: str | None = None
    query_caption_highlight_enabled: bool | None = None
    semantic_configuration_name: str | None = None
    select: list[str] | None = None
    skip: int | None = Field(default=None, ge=0)
    top: int = Field(default=20, ge=1, le=1000)
    scoring_statistics: str | None = None
    session_id: str | None = None
    vector_queries: list[VectorQuery] | None = None
    vector_filter_mode: str | None = None
    semantic_error_mode: str | None = None
    semantic_max_wait_in_milliseconds: int | None = None
    debug: str | None = None


SearchMode = Literal["bm25", "vector", "hybrid", "hybrid_semantic"]


class TextSearchRequest(StrictModel):
    query: str = Field(min_length=1)
    mode: SearchMode = "hybrid"
    vector_field: str = "content_vector"
    k: int = Field(default=50, ge=1, le=1000)
    exhaustive: bool = False
    semantic_configuration: str = "semantic-config"
    parameters: SearchOptions = Field(default_factory=SearchOptions)


class CompareRequest(TextSearchRequest):
    modes: list[SearchMode] = Field(
        default_factory=lambda: ["bm25", "vector", "hybrid"], min_length=1, max_length=4
    )
    repeats: int = Field(default=1, ge=1, le=5)
    seed: int = 42
    document_id_field: str = "document_id"
    qrels: dict[str, int] | None = None

    @model_validator(mode="after")
    def validate_comparison(self) -> "CompareRequest":
        if len(self.modes) != len(set(self.modes)):
            raise ValueError("modes must be unique")
        if not 10 <= self.parameters.top <= 50 or self.parameters.skip:
            raise ValueError("Comparison requires top 10–50 and no skip")
        if "hybrid_semantic" in self.modes and self.k < 50:
            raise ValueError("Semantic comparison requires k >= 50")
        if self.qrels and any(not 0 <= grade <= 3 for grade in self.qrels.values()):
            raise ValueError("qrels grades must be 0–3")
        return self


class AzureResponse(BaseModel):
    elapsed_ms: float
    azure: dict[str, Any]


class TextSearchResponse(AzureResponse):
    mode: SearchMode
    embedding_ms: float


class CompareResponse(BaseModel):
    query_embedding_ms: float
    runs: list[dict[str, Any]]
    mean_search_ms: dict[str, float]
