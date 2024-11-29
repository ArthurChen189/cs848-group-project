import os
import pandas as pd
from pathlib import Path
from realtabformer import REaLTabFormer

# # load the data
parent_df = pd.read_csv("rossmann-data/preprocessed/parent_table.csv")
child_df = pd.read_csv("rossmann-data/preprocessed/all_table.csv")
join_on = "Store"

# Make sure that the key columns in both the parent and the child table have the same name.
assert ((join_on in parent_df.columns) and
        (join_on in child_df.columns))

# Train the parent model.
# Non-relational or parent table. Don't include the unique_id field.
parent_model = REaLTabFormer(model_type="tabular")
parent_model.fit(parent_df.drop(join_on, axis=1))

# save the parent model
pdir = Path("rtf_parent/")
parent_model.save(pdir)

# Get the most recently saved parent model,
parent_model_path = pdir / "idXXX"
parent_model_path = sorted([
    p for p in pdir.glob("id*") if p.is_dir()],
    key=os.path.getmtime)[-1]

# load the child model
child_model = REaLTabFormer(
    model_type="relational",
    parent_realtabformer_path=parent_model_path,
    output_max_length=None,
    gradient_checkpointing=True,
    batch_size=2,
    gradient_accumulation_steps=4,
    train_size=0.8)

child_model.fit(
    df=child_df,
    in_df=parent_df,
    join_on=join_on)

# save the child model
pdir = Path("rtf_child/")
child_model.save(pdir)
