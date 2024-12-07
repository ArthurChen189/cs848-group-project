import sys
sys.path.insert(0, "/home/qfyan/cs848-group-project/REaLTabFormer/src")

import os
import pandas as pd
from pathlib import Path
import argparse
from realtabformer import REaLTabFormer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_df", type=str, help="Path to the parent table")
    parser.add_argument("--child_df", type=str, help="Path to the child table", required=True)
    parser.add_argument("--join_on", type=str, help="Column to join on")
    parser.add_argument("--output_dir", type=str, default="rtf_checkpoints", help="Directory to save the models")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    return parser.parse_args()

def main(args):
    # get the name of the output directory
    dataset_name = args.child_df.split("/")[-2]
    output_dir = Path(args.output_dir, dataset_name)
    if args.parent_df is None:
        # if no parent df is provided, that means we only need to train one model for non-relational data
        # load the data
        df = pd.read_csv(args.child_df)
        model = REaLTabFormer(model_type="tabular", gradient_accumulation_steps=4, batch_size=args.batch_size)
        model.fit(df)
    
        # save the model
        print(f"Saving model to {output_dir}")
        model.save(output_dir)

    else:
        # else, we need to train both the parent and the child models for relational data
        assert args.join_on is not None, "Join on column is required for relational data"
        assert args.parent_df is not None, "Parent df is required for relational data"

        # load the data
        parent_df = pd.read_csv(args.parent_df)
        child_df = pd.read_csv(args.child_df)
        join_on = args.join_on

        # Make sure that the key columns in both the parent and the child table have the same name.
        assert ((join_on in parent_df.columns) and
                (join_on in child_df.columns))

        # Train the parent model.
        # Non-relational or parent table. Don't include the unique_id field.
        parent_model = REaLTabFormer(model_type="tabular", batch_size=args.batch_size)
        parent_model.fit(parent_df.drop(join_on, axis=1))

        # save the parent model
        pdir = Path(output_dir, "parent")
        parent_model.save(pdir)

        # Get the most recently saved parent model,
        parent_model_path = pdir / "idXXX"
        parent_model_path = sorted([
            p for p in pdir.glob("id*") if p.is_dir()],
            key=os.path.getmtime)[-1]
        print(f"Using parent model from {parent_model_path}")

        # load the child model
        child_model = REaLTabFormer(
            model_type="relational",
            parent_realtabformer_path=parent_model_path,
            output_max_length=None,
            gradient_checkpointing=True,
            batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            train_size=0.8)

        child_model.fit(
            df=child_df,
            in_df=parent_df,
            join_on=join_on)

        # save the child model
        cdir = Path(output_dir, "child")
        child_model.save(cdir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
