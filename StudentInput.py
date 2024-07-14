from pydantic import BaseModel, Field
from typing import Optional

class StudentInfo(BaseModel):
    Marital_status: Optional[int] = Field(..., alias="Marital status")
    Application_mode: Optional[int] = Field(..., alias="Application mode")
    Application_order: Optional[int] = Field(..., alias="Application order")
    Course: Optional[int] = Field(..., alias="Course")
    Daytime_evening_attendance: Optional[int] = Field(..., alias="Daytime/evening attendance")
    Previous_qualification: Optional[int] = Field(..., alias="Previous qualification")
    Previous_qualification_grade: Optional[float] = Field(..., alias="Previous qualification (grade)")
    Nacionality: Optional[int] = Field(..., alias="Nacionality")
    Mothers_qualification: Optional[int] = Field(..., alias="Mother's qualification")
    Fathers_qualification: Optional[int] = Field(..., alias="Father's qualification")
    Mothers_occupation: Optional[int] = Field(..., alias="Mother's occupation")
    Fathers_occupation: Optional[int] = Field(..., alias="Father's occupation")
    Admission_grade: Optional[float] = Field(..., alias="Admission grade")
    Displaced: Optional[int] = Field(..., alias="Displaced")
    Educational_special_needs: Optional[int] = Field(..., alias="Educational special needs")
    Debtor: Optional[int] = Field(..., alias="Debtor")
    Tuition_fees_up_to_date: Optional[int] = Field(..., alias="Tuition fees up to date")
    Gender: Optional[int] = Field(..., alias="Gender")
    Scholarship_holder: Optional[int] = Field(..., alias="Scholarship holder")
    Age_at_enrollment: Optional[int] = Field(..., alias="Age at enrollment")
    International: Optional[int] = Field(..., alias="International")
    Curricular_units_1st_sem_credited: Optional[int] = Field(..., alias="Curricular units 1st sem (credited)")
    Curricular_units_1st_sem_enrolled: Optional[int] = Field(..., alias="Curricular units 1st sem (enrolled)")
    Curricular_units_1st_sem_evaluations: Optional[int] = Field(..., alias="Curricular units 1st sem (evaluations)")
    Curricular_units_1st_sem_approved: Optional[int] = Field(..., alias="Curricular units 1st sem (approved)")
    Curricular_units_1st_sem_grade: Optional[float] = Field(..., alias="Curricular units 1st sem (grade)")
    Curricular_units_1st_sem_without_evaluations: Optional[int] = Field(..., alias="Curricular units 1st sem (without evaluations)")
    Curricular_units_2nd_sem_credited: Optional[int] = Field(..., alias="Curricular units 2nd sem (credited)")
    Curricular_units_2nd_sem_enrolled: Optional[int] = Field(..., alias="Curricular units 2nd sem (enrolled)")
    Curricular_units_2nd_sem_evaluations: Optional[int] = Field(..., alias="Curricular units 2nd sem (evaluations)")
    Curricular_units_2nd_sem_approved: Optional[int] = Field(..., alias="Curricular units 2nd sem (approved)")
    Curricular_units_2nd_sem_grade: Optional[float] = Field(..., alias="Curricular units 2nd sem (grade)")
    Curricular_units_2nd_sem_without_evaluations: Optional[int] = Field(..., alias="Curricular units 2nd sem (without evaluations)")
    Unemployment_rate: Optional[float] = Field(..., alias="Unemployment rate")
    Inflation_rate: Optional[float] = Field(..., alias="Inflation rate")
    GDP: Optional[float] = Field(..., alias="GDP")

    class Config:
        # allow_population_by_field_name = True
        populate_by_name=True

