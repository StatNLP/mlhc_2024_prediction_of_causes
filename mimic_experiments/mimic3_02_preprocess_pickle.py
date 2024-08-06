import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np


def f(x):
    global z_err  # loss where HADM_ID is empty
    chart_time = x.CHARTTIME
    try:
        for icu_times in x.icustay_times:
            if icu_times[1] <= chart_time <= icu_times[2]:
                return icu_times[0]
    except TypeError:
        z_err += 1


# Get ts_ind.
def inv_list(x, start=0):
    d = {}
    for i in range(len(x)):
        d[x[i]] = i
    return d


# Read extracted time series data.
mimic_data_dir = "../mimic-iii-clinical-database-1.4/"
events = pd.read_csv(
    "mimic_iii_events.csv",
    usecols=["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"],
)
icu = pd.read_csv("mimic_iii_icu.csv")

# Convert times to type datetime.
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
icu.INTIME = pd.to_datetime(icu.INTIME)
icu.OUTTIME = pd.to_datetime(icu.OUTTIME)

# Assign ICUSTAY_ID to rows without it. Remove rows that can't be assigned one.
icu["icustay_times"] = icu.apply(lambda x: [x.ICUSTAY_ID, x.INTIME, x.OUTTIME], axis=1)
adm_icu_times = icu.groupby("HADM_ID").agg({"icustay_times": list}).reset_index()
icu.drop(columns=["icustay_times"], inplace=True)
events = events.merge(adm_icu_times, on=["HADM_ID"], how="left")
idx = events.ICUSTAY_ID.isna()
tqdm.pandas()

z_err = 0


events.loc[idx, "ICUSTAY_ID"] = (events.loc[idx]).progress_apply(f, axis=1)
events.drop(columns=["icustay_times"], inplace=True)
events = events.loc[events.ICUSTAY_ID.notna()]
events.drop(columns=["HADM_ID"], inplace=True)
print("z_err is", z_err)

# Filter icu table.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

# Get rel_charttime in minutes.
events = events.merge(icu[["ICUSTAY_ID", "INTIME"]], on="ICUSTAY_ID", how="left")
events["rel_charttime"] = events.CHARTTIME - events.INTIME
events.drop(columns=["INTIME", "CHARTTIME"], inplace=True)
events.rel_charttime = events.rel_charttime.dt.total_seconds() // 60

# Save current icu table.
icu_full = icu.copy()

# Get icustays which lasted for atleast 24 hours.
icu = icu.loc[(icu.OUTTIME - icu.INTIME) >= pd.Timedelta(24, "h")]

# Get icustays with patient alive for atleast 24 hours.
adm = pd.read_csv(mimic_data_dir + "ADMISSIONS.csv", usecols=["HADM_ID", "DEATHTIME"])
icu = icu.merge(adm, on="HADM_ID", how="left")
icu.DEATHTIME = pd.to_datetime(icu.DEATHTIME)
icu = icu.loc[
    ((icu.DEATHTIME - icu.INTIME) >= pd.Timedelta(24, "h")) | icu.DEATHTIME.isna()
]

# Get icustays with aleast one event in first 24h.
icu = icu.loc[
    icu.ICUSTAY_ID.isin(events.loc[events.rel_charttime < 24 * 60].ICUSTAY_ID)
]

# Get sup and unsup icustays.
all_icustays = np.array(icu_full.ICUSTAY_ID)
sup_icustays = np.array(icu.ICUSTAY_ID)
unsup_icustays = np.setdiff1d(all_icustays, sup_icustays)
all_icustays = np.concatenate((sup_icustays, unsup_icustays), axis=-1)


icustay_to_ind = inv_list(all_icustays)
events["ts_ind"] = events.ICUSTAY_ID.map(icustay_to_ind)

# Rename some columns.
events.rename(
    columns={"rel_charttime": "minute", "NAME": "variable", "VALUENUM": "value"},
    inplace=True,
)

# Add gender and age.
icu_full["ts_ind"] = icu_full.ICUSTAY_ID.map(icustay_to_ind)
data_age = icu_full[["ts_ind", "AGE"]]
data_age["variable"] = "Age"
data_age.rename(columns={"AGE": "value"}, inplace=True)
data_gen = icu_full[["ts_ind", "GENDER"]]
data_gen.loc[data_gen.GENDER == "M", "GENDER"] = 0
data_gen.loc[data_gen.GENDER == "F", "GENDER"] = 1
data_gen["variable"] = "Gender"
data_gen.rename(columns={"GENDER": "value"}, inplace=True)
data = pd.concat((data_age, data_gen), ignore_index=True)
data["minute"] = 0
events = pd.concat((data, events), ignore_index=True)

# Drop duplicate events.
events.drop_duplicates(inplace=True)

# Add mortality label.
adm = pd.read_csv(
    mimic_data_dir + "ADMISSIONS.csv", usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]
)
oc = icu_full[["ts_ind", "HADM_ID", "SUBJECT_ID"]].merge(adm, on="HADM_ID", how="left")
oc = oc.rename(columns={"HOSPITAL_EXPIRE_FLAG": "in_hospital_mortality"})

# Get train-valid-test split for sup task.
all_sup_subjects = icu.SUBJECT_ID.unique()
np.random.seed(0)
np.random.shuffle(all_sup_subjects)
S = len(all_sup_subjects)
bp1, bp2 = int(0.64 * S), int(0.8 * S)
train_sub = all_sup_subjects[:bp1]
valid_sub = all_sup_subjects[bp1:bp2]
test_sub = all_sup_subjects[bp2:]
icu["ts_ind"] = icu.ICUSTAY_ID.map(icustay_to_ind)
train_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(train_sub)].ts_ind)
valid_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(valid_sub)].ts_ind)
test_ind = np.array(icu.loc[icu.SUBJECT_ID.isin(test_sub)].ts_ind)

# Filter columns.
events = events[["ts_ind", "minute", "variable", "value", "TABLE"]]

# Convert minute to hour.
events["hour"] = events["minute"] / 60
events.drop(columns=["minute"], inplace=True)

# Aggregate data.
events["value"] = events["value"].astype(float)
events.loc[events["TABLE"].isna(), "TABLE"] = "N/A"
events = (
    events.groupby(["ts_ind", "hour", "variable"])
    .agg({"value": "mean", "TABLE": "unique"})
    .reset_index()
)


# Second definition of x - for Table
def f(x):
    if len(x) == 0:
        return x[0]
    else:
        return ",".join(x)


events["TABLE"] = events["TABLE"].apply(f)

# Save data.
pickle.dump(
    [events, oc, train_ind, valid_ind, test_ind],
    open("mimic_iii_preprocessed.pkl", "wb"),
)

# Normalize data and save.
ts, oc, train_ind, valid_ind, test_ind = pickle.load(
    open("mimic_iii_preprocessed.pkl", "rb")
)
means_stds = ts.groupby("variable").agg({"value": ["mean", "std"]})
means_stds.columns = [col[1] for col in means_stds.columns]
means_stds.loc[means_stds["std"] == 0, "std"] = 1
ts = ts.merge(means_stds.reset_index(), on="variable", how="left")
ii = ~ts.variable.isin(["Age", "Gender"])
ts.loc[ii, "value"] = (ts.loc[ii, "value"] - ts.loc[ii, "mean"]) / ts.loc[ii, "std"]
pickle.dump(
    [ts, oc, train_ind, valid_ind, test_ind], open("mimic_iii_preprocessed.pkl", "wb")
)
