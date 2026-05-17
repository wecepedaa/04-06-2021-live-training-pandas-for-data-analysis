import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from abc import ABC, abstractmethod

# ========================== CONSTANTS ==========================
_SNAPSHOT_DATE = pd.to_datetime('2019-09-30')

_CLAIM_STATUS_MAP: Dict[str, int] = {
    "open": 0,
    "closed": 1,
    "re-open": 2,
    "reopen": 2,
}

# ========================== INTERFACES ==========================
class FeatureTransformer(ABC):
    """Interface para todos los transformadores de features"""
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# ========================== FEATURE ENGINEER ==========================
class ClaimFeatureEngineer:
    """
    Clase principal responsable de generar todas las features.
    Cumple con SRP y OCP.
    """
    
    def __init__(self, 
                 provider_encoder: Any = None,   # CategoricalEncoder inyectado
                 snapshot_date: pd.Timestamp = _SNAPSHOT_DATE):
        self.provider_encoder = provider_encoder
        self.snapshot_date = snapshot_date

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todas las transformaciones de forma vectorizada"""
        df = df.copy()
        
        self._create_ratios(df)
        self._create_binary_flags(df)
        self._create_log_features(df)
        self._create_date_features(df)
        self._create_encoded_features(df)
        
        return df

    # ====================== MÉTODOS PRIVADOS ======================
    
    def _create_ratios(self, df: pd.DataFrame):
        """Crea todas las ratios de forma vectorizada"""
        total = df.get('IncurredTotal', pd.Series(0, index=df.index)).fillna(0).astype(float)
        
        df['Medical_Ratio']   = np.divide(df.get('IncurredMedical', 0).fillna(0), total, 
                                         where=total != 0, out=np.zeros_like(total))
        df['Indemnity_Ratio'] = np.divide(df.get('IncurredIndemnity', 0).fillna(0), total, 
                                         where=total != 0, out=np.zeros_like(total))
        df['Expense_Ratio']   = np.divide(df.get('IncurredExpense', 0).fillna(0), total, 
                                         where=total != 0, out=np.zeros_like(total))
        
        df['Ratios_Sum'] = df['Medical_Ratio'] + df['Indemnity_Ratio'] + df['Expense_Ratio']

    def _create_binary_flags(self, df: pd.DataFrame):
        """Crea todos los flags binarios"""
        # In MPN
        df['In_MPN_Flag'] = pd.to_numeric(
            df.get('NetworkStatus', 0), errors='coerce'
        ).clip(0, 1).fillna(0).astype(int)
        
        # Has PTP
        df['Has_PTP'] = pd.to_numeric(
            df.get('ClaimHasPTP', 0), errors='coerce'
        ).clip(0, 1).fillna(0).astype(int)
        
        # Is Litigated
        df['Is_Litigated'] = (
            df.get('LitigationFlag', "")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq('litigated')
            .astype(int)
        )

    def _create_log_features(self, df: pd.DataFrame):
        """Crea features logarítmicas"""
        df['TTDDays_log'] = np.log1p(
            df.get('TTDDays', 0).fillna(0).clip(lower=0).astype(float)
        )
        
        df['IncurredTotal_log'] = np.log1p(
            df.get('IncurredTotal', 0).fillna(0).clip(lower=0).astype(float)
        )

    def _create_date_features(self, df: pd.DataFrame):
        """Crea features basadas en fechas"""
        injury_date = pd.to_datetime(df.get('DateOfInjury'), errors='coerce')
        
        # Claim Age
        df['ClaimAge_Days'] = (self.snapshot_date - injury_date).dt.days.clip(lower=0).fillna(0).astype(int)
        
        # Days to Attorney (solo si está litigado)
        rep_date = pd.to_datetime(df.get('LegalRepresentationDate'), errors='coerce')
        days_to_attorney = (rep_date - injury_date).dt.days
        df['Days_To_Attorney'] = np.where(
            (df['Is_Litigated'] == 1) & days_to_attorney.notna(),
            days_to_attorney.astype(int),
            -1
        )
        
        # Days to MMIPS
        mmips_date = pd.to_datetime(df.get('MMIPSDate'), errors='coerce')
        days_to_mmips = (mmips_date - injury_date).dt.days
        df['Days_To_MMIPS'] = np.where(
            days_to_mmips.notna() & mmips_date.notna() & injury_date.notna(),
            days_to_mmips.astype(int),
            -1
        )

    def _create_encoded_features(self, df: pd.DataFrame):
        """Crea features codificadas"""
        # Claim Status
        status = (df.get('ClaimStatus', "")
                  .astype(str)
                  .str.strip()
                  .str.lower())
        df['ClaimStatus_Encoded'] = status.map(_CLAIM_STATUS_MAP).fillna(0).astype(int)
        
        # Provider Specialty (si se inyectó encoder)
        if self.provider_encoder is not None and 'ProviderSpecialty' in df.columns:
            specialties = df['ProviderSpecialty'].fillna("").astype(str)
            df['ProviderSpecialty_Encoded'] = self.provider_encoder.transform(specialties)
        else:
            df['ProviderSpecialty_Encoded'] = 0

        # Provider Score
        df['Provider_Score'] = pd.to_numeric(
            df.get('Score', 0), errors='coerce'
        ).clip(0, 5).fillna(0).astype(int)

   