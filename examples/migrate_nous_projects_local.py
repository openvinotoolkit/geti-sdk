import json
import os
import sys
from collections import Counter
from json import JSONDecodeError

from tqdm import tqdm

from sc_api_tools import SCRESTClient
from sc_api_tools.nous.nous2sc import migrate_nous_chain
from sc_api_tools.rest_clients import ProjectClient

if __name__ == "__main__":
    # unzip https://intel-my.sharepoint.com/:u:/p/laurens_hogeweg/EWVlj9qX2X1EuPT91GBemkkB4TlYAjg6jK02XGVM-n_2lw?e=ibdiwY
    ann_dir = "/home/lhogeweg/Documents/Datasets/diopsis21clean/annotation"
    labels = []
    ann_files = os.listdir(ann_dir)
    specific_images = None
    for fn in ann_files:
        try:
            j_ann = json.load(open(os.path.join(ann_dir, fn)))
            for j_shape in j_ann["data"]:
                image_labels = [x["name"] for x in j_shape["labels"]]
                # if "Muscidae" in image_labels:
                #     specific_images.append(fn.split(".")[0])
                labels.extend(image_labels)
        except JSONDecodeError:
            print(fn)
    # exit(0)

    counts = Counter(labels).most_common()
    print(counts)
    sufficient_labels = set([x[0] for x in counts if x[1] >= 6])
    # exit(0)
    sufficient_labels -= {"Cataclysta lemnata - do not use", "Muscidae"}
    print(sufficient_labels)

    update_annotations = True
    if update_annotations:
        for fn in tqdm(ann_files, total=len(ann_files)):
            try:
                p = os.path.join(ann_dir, fn)
            except JSONDecodeError:
                print(fn)
                raise

            j_ann = json.load(open(p))
            for j_shape in j_ann["data"]:
                j_shape["labels"] = [
                    x for x in j_shape["labels"] if x["name"] in sufficient_labels
                ]
            json.dump(j_ann, open(p, "w"), indent=2)

    # from https://intel-my.sharepoint.com/:u:/p/laurens_hogeweg/EWsuZUo9d31GlYTV4e6u1sIBn8xwGDXqS5rggbx9oOrnyA?e=cGpSDL
    j_project = json.load(open("/mnt/big/diopsis_s/project.json"))
    id_to_label = {}
    for j_label in j_project["pipeline"]["tasks"][3]["properties"][0]["user_value"]:
        print(j_label)
        print(j_label["name"], j_label["parent"], j_label["id"])
        id_to_label[j_label["id"]] = j_label

    labels_migrate = []

    print("|LABELS|", len(id_to_label))

    for id_, label in id_to_label.items():
        label_name = label["name"]
        if (
            label_name not in sufficient_labels
            or label_name == "Object"
            or "Empty" in label_name
        ):
            continue
        migrate_label = {
            "name": label_name,
        }
        if label["parent"] in id_to_label and label_name != "Empty Classification":
            migrate_label["parent_id"] = id_to_label[label["parent"]]["name"]
            migrate_label["group"] = "g_" + id_to_label[label["parent"]]["name"]
            print("migrate_label['group']", migrate_label["group"])
        labels_migrate.append(migrate_label)
    labels_migrate.append({"name": "No Animalia"})
    print(labels_migrate)

    client = SCRESTClient(
        host="https://sc-demo.iotg.sclab.intel.com/",
        username="laurens.hogeweg@intel.com",
        password=sys.argv[1],
    )




    migrate_nous_chain(
        rest_client=client,
        export_path="/home/lhogeweg/Documents/Datasets/diopsis21.zip",
        # https://intel-my.sharepoint.com/:u:/p/laurens_hogeweg/EWVlj9qX2X1EuPT91GBemkkB4TlYAjg6jK02XGVM-n_2lw?e=ibdiwY
        # unzip this once to temp_dir to speed up the script if run multiple times
        task_types=["detection", "classification"],
        labels_per_task=[["Object"], labels_migrate],
        project_name="diopsis",
        temp_dir="/home/lhogeweg/Documents/Datasets/diopsis21clean",
        offset=0,  # start uploading annotations from this offset
        specific_images=specific_images,
    )
