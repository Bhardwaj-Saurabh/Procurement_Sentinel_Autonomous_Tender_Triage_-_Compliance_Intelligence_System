"""Tender-related Pydantic schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field


class TenderValue(BaseModel):
    """Tender monetary value."""

    amount: Optional[Decimal] = None
    currency: str = "GBP"


class TenderPeriod(BaseModel):
    """Tender submission period."""

    start_date: Optional[datetime] = Field(None, alias="startDate")
    end_date: Optional[datetime] = Field(None, alias="endDate")

    class Config:
        populate_by_name = True


class Buyer(BaseModel):
    """Tender buyer/organization."""

    name: Optional[str] = None
    id: Optional[str] = None


class TenderItem(BaseModel):
    """Item within a tender."""

    id: str
    description: Optional[str] = None
    cpv_code: Optional[str] = Field(None, alias="classification")


class TenderDocument(BaseModel):
    """Document attached to tender."""

    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    format: Optional[str] = None


class Tender(BaseModel):
    """Main tender model."""

    ocid: str
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    value: Optional[TenderValue] = None
    tender_period: Optional[TenderPeriod] = Field(None, alias="tenderPeriod")
    buyer: Optional[Buyer] = None
    items: list[TenderItem] = Field(default_factory=list)
    documents: list[TenderDocument] = Field(default_factory=list)
    published_date: Optional[datetime] = None

    class Config:
        populate_by_name = True


class TenderCreate(BaseModel):
    """Schema for creating a tender in DB."""

    ocid: str
    title: Optional[str] = None
    description: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_id: Optional[str] = None
    value_amount: Optional[Decimal] = None
    value_currency: str = "GBP"
    deadline_date: Optional[datetime] = None
    raw_data: Optional[dict] = None
