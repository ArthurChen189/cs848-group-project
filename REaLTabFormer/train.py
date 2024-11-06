import os
import pandas as pd
from pathlib import Path
from realtabformer import REaLTabFormer

# load the data
parent_df = pd.read_csv("rossmann-data/preprocessed/parent_table.csv")
child_df = pd.read_csv("rossmann-data/preprocessed/all_table.csv")
join_on = "Store"

# Make sure that the key columns in both the parent and the child table have the same name.
assert ((join_on in parent_df.columns) and
        (join_on in child_df.columns))

# Non-relational or parent table. Don't include the unique_id field.
parent_model = REaLTabFormer(model_type="tabular")
parent_model.fit(parent_df.drop(join_on, axis=1))

pdir = Path("rtf_parent/")
parent_model.save(pdir)

# # Get the most recently saved parent model,
# # or a specify some other saved model.
# parent_model_path = pdir / "idXXX"
parent_model_path = sorted([
    p for p in pdir.glob("id*") if p.is_dir()],
    key=os.path.getmtime)[-1]

# load the child model
child_model = REaLTabFormer(
    model_type="relational",
    parent_realtabformer_path=parent_model_path,
    output_max_length=None,
    train_size=0.8)

child_model.fit(
    df=child_df,
    in_df=parent_df,
    join_on=join_on)

# Generate parent samples.
parent_samples = parent_model.sample(len(parent_df))

# Create the unique ids based on the index.
parent_samples.index.name = join_on
parent_samples = parent_samples.reset_index()

# Generate the relational observations.
child_samples = child_model.sample(
    input_unique_ids=parent_samples[join_on],
    input_df=parent_samples.drop(join_on, axis=1),
    gen_batch=64)

# Save the samples.
parent_samples.to_csv("synthesized/parent_samples.csv", index=False)
child_samples.to_csv("synthesized/child_samples.csv", index=False)
