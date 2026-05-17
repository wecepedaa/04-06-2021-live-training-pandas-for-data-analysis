import pandas as pd
import numpy as np
from datetime import datetime

# ====================== IMPORTAMOS NUESTRA CLASE ======================
from claim_feature_engineer import ClaimFeatureEngineer   # Asegúrate de tener este archivo

# ========================== MAIN ==========================
def main():
    print("🚀 Iniciando Feature Engineering...\n")
    
    # 1. Cargar el DataFrame desde CSV
    csv_path = "metricsClaims.csv"   # Cambia esto por tu archivo real
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Archivo cargado correctamente: {df.shape[0]:,} filas y {df.shape[1]} columnas")
    except FileNotFoundError:
        print(f"⚠️  Archivo {csv_path} no encontrado. Creando un dataset de prueba...")
        df = create_sample_data()
        df.to_csv(csv_path, index=False)
        print(f"✅ Dataset de prueba creado y guardado como '{csv_path}'")
    
    # 2. Instanciar el Feature Engineer
    engineer = ClaimFeatureEngineer()
    
    # 3. Aplicar todas las transformaciones
    print("🔄 Aplicando transformaciones vectorizadas...")
    df_transformed = engineer.transform(df)
    
    # 4. Resultados
    print(f"\n✅ Transformación completada!")
    print(f"   Filas: {df_transformed.shape[0]:,}")
    print(f"   Columnas nuevas: {df_transformed.shape[1] - df.shape[1]}")
    
    # Mostrar algunas columnas creadas
    new_columns = [col for col in df_transformed.columns if col not in df.columns]
    print("\n📋 Nuevas features creadas:")
    for col in sorted(new_columns):
        print(f"   → {col}")
    
    # Mostrar ejemplo de datos
    print("\n📊 Ejemplo de las primeras 5 filas (algunas columnas):")
    cols_to_show = ['IncurredTotal', 'Medical_Ratio', 'Indemnity_Ratio', 
                   'Is_Litigated', 'ClaimAge_Days', 'Days_To_Attorney', 'Provider_Score']
    cols_to_show = [col for col in cols_to_show if col in df_transformed.columns]
    
    print(df_transformed[cols_to_show].head())
    
    # Guardar resultado
    output_path = "claims_data_with_features.csv"
    df_transformed.to_csv(output_path, index=False)
    print(f"\n💾 Resultado guardado en: {output_path}")


def create_sample_data() -> pd.DataFrame:
    """Crea un dataset de prueba con las columnas necesarias"""
    np.random.seed(42)
    n = 50_000  # Puedes cambiar a 100000 o más para probar rendimiento
    
    data = {
        'IncurredTotal': np.random.lognormal(8, 1.5, n).astype(int),
        'IncurredMedical': np.random.lognormal(7.5, 1.6, n).astype(int),
        'IncurredIndemnity': np.random.lognormal(7, 1.7, n).astype(int),
        'IncurredExpense': np.random.lognormal(6, 1.4, n).astype(int),
        'TTDDays': np.random.randint(0, 730, n),
        'NetworkStatus': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'ClaimHasPTP': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'LitigationFlag': np.random.choice(['Litigated', 'Not Litigated', ''], n, p=[0.15, 0.75, 0.1]),
        'DateOfInjury': pd.date_range(start='2018-01-01', periods=n, freq='D').astype(str),
        'LegalRepresentationDate': pd.date_range(start='2018-02-01', periods=n, freq='D').astype(str),
        'MMIPSDate': pd.date_range(start='2019-01-01', periods=n, freq='D').astype(str),
        'ClaimStatus': np.random.choice(['Open', 'Closed', 'Re-Open', 'closed', 'REOPEN'], n),
        'Score': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.3, 0.1, 0.15, 0.25, 0.15, 0.05]),
        'ProviderSpecialty': np.random.choice(['Orthopedic', 'Physical Therapy', 'Neurology', 
                                              'General Medicine', ''], n),
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()