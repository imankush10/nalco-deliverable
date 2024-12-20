from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Ensure Base is imported or defined properly
Base = declarative_base()

DATABASE_URL = "sqlite:///./real.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class RealTimeData(Base):
    __tablename__ = 'real_time_data_prediction'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    EMUL_OIL_L_TEMP_PV_VAL0 = Column(Float)
    STAND_OIL_L_TEMP_PV_REAL_VAL0 = Column(Float)
    GEAR_OIL_L_TEMP_PV_REAL_VAL0 = Column(Float)
    EMUL_OIL_L_PR_VAL0 = Column(Float)
    ROD_DIA_MM_VAL0 = Column(Float)
    QUENCH_CW_FLOW_EXIT_VAL0 = Column(Float)
    CAST_WHEEL_RPM_VAL0 = Column(Float)
    BAR_TEMP_VAL0 = Column(Float)
    QUENCH_CW_FLOW_ENTRY_VAL0 = Column(Float)
    GEAR_OIL_L_PR_VAL0 = Column(Float)
    STANDS_OIL_L_PR_VAL0 = Column(Float)
    TUNDISH_TEMP_VAL0 = Column(Float)
    RM_MOTOR_COOL_WATER__VAL0 = Column(Float)
    ROLL_MILL_AMPS_VAL0 = Column(Float)
    RM_COOL_WATER_FLOW_VAL0 = Column(Float)
    EMULSION_LEVEL_ANALO_VAL0 = Column(Float)
    furnaceTemp = Column(Float)
    SI = Column(Float)
    FE = Column(Float)
    TI = Column(Float)
    V = Column(Float)
    MN = Column(Float)
    OTHIMP = Column(Float)
    AL = Column(Float)
    Uts = Column(Float)
    Elongation = Column(Float)
    Conductivity = Column(Float)

Base.metadata.create_all(bind=engine)
