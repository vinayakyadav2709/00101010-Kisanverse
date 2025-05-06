# models.py
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime
import logging

log = logging.getLogger(__name__)

# --- Agronomic Data Structure ---
class AgronomicInfo(BaseModel):
    crop_name: str
    suitable_soils: List[str] = Field(default_factory=list)
    water_needs: str = Field(default="Unknown", description="General water requirement category (e.g., Low, Moderate, High, Very High)")
    input_cost_cat: str = Field(default="Medium", description="Category of input cost (e.g., Low, Medium, High)")
    fertilizer_needs: str = Field(default="Standard NPK", description="General fertilizer requirements or specific notes")
    common_pests: List[str] = Field(default_factory=list, description="List of common pests and diseases")
    typical_yield_range_q_acre: Optional[Tuple[float, float]] = Field(default=None, description="Typical yield range in quintals per acre (low, high)")
    general_notes: Optional[str] = Field(default=None, description="Other relevant notes (e.g., duration, sensitivities)")

    @field_validator('typical_yield_range_q_acre')
    @classmethod
    def check_yield_range(cls, v: Optional[Tuple[float, float]]):
        if v is not None:
            if not isinstance(v, tuple) or len(v) != 2:
                raise ValueError("typical_yield_range_q_acre must be a tuple of two floats")
            if not all(isinstance(x, (int, float)) for x in v):
                 raise ValueError("Yield range values must be numeric")
            if v[0] < 0 or v[1] < 0:
                raise ValueError("Yield range values cannot be negative")
            if v[0] > v[1]:
                raise ValueError("Yield range: low value cannot be greater than high value")
        return v

# --- Input Models ---
class WeatherInfo(BaseModel):
    date: date
    temp_max: float = Field(..., description="Maximum temperature (°C)")
    temp_min: float = Field(..., description="Minimum temperature (°C)")
    precip: float = Field(..., description="Total precipitation (mm)")
    wind_max: Optional[float] = Field(default=None, description="Maximum wind speed (km/h)")
    radiation: Optional[float] = Field(default=None, description="Surface shortwave radiation sum (MJ/m²)")

class SubsidyInfo(BaseModel):
    # Fields extracted from ss.json (ignoring others)
    type: str = Field(default="Unknown", description="Type of subsidy (e.g., financial, asset, input)")
    provider: str = Field(default="Unknown", description="Providing agency/department")
    program: str = Field(..., description="Name of the subsidy program/scheme")
    description: Optional[str] = Field(default=None)
    eligibility: Optional[str] = Field(default=None)
    benefits: str = Field(..., description="Description of benefits provided")
    locations: List[str] = Field(default_factory=list, description="List of applicable location codes/names (if any)")
    value_estimate_inr: Optional[float] = Field(default=None, description="Estimated monetary value in INR (if available)")

class InputParams(BaseModel):
    soil_type: str = Field(..., description="Detected or user-provided soil type")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    planting_date_estimate: date = Field(..., description="Estimated planting start date")
    land_size_acres: float = Field(..., gt=0, description="Size of the land plot in acres")
    weather_forecast: List[WeatherInfo] = Field(default_factory=list, description="List of daily weather forecast data")
    available_subsidies: List[SubsidyInfo] = Field(default_factory=list, description="List of potentially available subsidies")

    @field_validator('weather_forecast')
    @classmethod
    def check_weather_forecast_list(cls, v):
        if not isinstance(v, list):
             raise ValueError("weather_forecast must be a list")
        return v

# --- Output Models ---
class RequestDetails(BaseModel):
    latitude: float
    longitude: float
    soil_type: str
    land_size_acres: float
    planting_date_estimate: date
    timestamp: datetime = Field(default_factory=datetime.now)

class ExplanationPoint(BaseModel):
    reason_type: str = Field(..., description="Category of the explanation (e.g., Profitability, Agronomic)")
    detail: str = Field(..., description="Specific explanation text")
    ai_component: str = Field(..., description="Underlying AI technique involved (e.g., Rule-Based, Predictive)")

class KeyMetrics(BaseModel):
    expected_yield_range: str = Field(..., description="Estimated yield range string (e.g., '10.0-15.0 q/acre')")
    price_forecast_trend: str = Field(..., description="Simulated price trend (e.g., Rising, Stable)")
    estimated_input_cost_category: str = Field(..., description="Input cost category (e.g., Low, Medium)")
    primary_fertilizer_needs: str = Field(..., description="Summary of fertilizer needs")

class RelevantSubsidyOutput(BaseModel):
    program: str
    provider: str
    benefit_summary: str
    estimated_value_inr: Optional[float] = None

# --- MODIFIED: Specific Plotting Data Point Models ---
class PriceDataPoint(BaseModel):
    date: date # Date is required for price points
    predicted_price_min: float
    predicted_price_max: float

class WaterDataPoint(BaseModel):
    growth_stage: str # Stage is required for water points
    relative_need_level: int # Level is required

class FertilizerDataPoint(BaseModel):
    stage: str # Stage is required for fertilizer points
    timing: str
    nutrients: str
# --- END MODIFIED ---

# --- MODIFIED: Specific Chart Data Models ---
class PriceChartData(BaseModel):
    description: str = Field(..., description="Title or description for the price chart")
    data: List[PriceDataPoint] = Field(default_factory=list)

class WaterChartData(BaseModel):
    description: str = Field(..., description="Title or description for the water chart")
    data: List[WaterDataPoint] = Field(default_factory=list)

class FertilizerChartData(BaseModel):
    description: str = Field(..., description="Title or description for the fertilizer chart")
    data: List[FertilizerDataPoint] = Field(default_factory=list)
# --- END MODIFIED ---

# --- MODIFIED: PlottingData uses specific chart types ---
class PlottingData(BaseModel):
    price_forecast_chart: PriceChartData
    water_need_chart: WaterChartData
    fertilizer_schedule_chart: FertilizerChartData
# --- END MODIFIED ---

class PesticideSuggestion(BaseModel):
    chemical_name: str
    target_pest: str = Field(..., description="Pest/disease targeted, reason (e.g., 'bollworm (common)', 'blight (alert)')")
    timing_stage: str = Field(..., description="Recommended application timing/stage")

class NewsItemReference(BaseModel):
    """Represents a relevant news snippet identified by RAG/LLM."""
    headline: str
    url: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None # Date as string from source/DB (ISO format preferred)

# --- Recommendation ---
class Recommendation(BaseModel):
    rank: int
    crop_name: str
    recommendation_score: float = Field(..., description="Normalized score (0-100)")
    explanation_points: List[ExplanationPoint] = Field(default_factory=list)
    key_metrics: KeyMetrics
    relevant_subsidies: List[RelevantSubsidyOutput] = Field(default_factory=list)
    primary_risks: List[str] = Field(default_factory=list)
    suggested_pesticides: List[PesticideSuggestion] = Field(default_factory=list)
    plotting_data: PlottingData # Uses the updated PlottingData model
    relevant_news: List[NewsItemReference] = Field(default_factory=list, description="List of relevant news items identified for this crop")

# --- Final Response ---
class RecommendationResponse(BaseModel):
    request_details: RequestDetails
    recommendations: List[Recommendation] = Field(default_factory=list)
    weather_context_summary: str = Field(..., description="Summary of the weather forecast's implications")
    news_headlines_considered: List[str] = Field(default_factory=list, description="Headlines retrieved via RAG and considered by the LLM.")
    news_impact_summary_llm: Optional[str] = Field(default=None, description="LLM-generated summary of the news context's impact.")
    overall_llm_summary: Optional[str] = Field(default=None, description="Final user-facing summary generated by LLM.")

    class Config:
        # Configuration for Pydantic model behavior if needed
        pass