"""
Convert the official Oxford-IIIT Pet files to COCO JSON for DEIM.

Expected layout (official images.tar.gz + annotations.tar.gz unpacked into one folder):
    <root>/images/*.jpg
    <root>/annotations/trainval.txt
    <root>/annotations/xmls/*.xml

Class labels come from the breed class id in trainval.txt (1-37), NOT from the
XML <name> tag, which is only 'cat' / 'dog'. COCO category_id is stored 0-based
(class_id - 1, i.e. 0..36) because DEIM with remap_mscoco_category=False uses
category_id directly as the label index, and index == num_classes is background.

Usage:
    python tools/pets_to_coco.py --root data/oxford_pets --out data/oxford_pets/coco
"""

import argparse
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from PIL import Image


def breed_from_name(name):
    # 'Bengal_105' -> 'Bengal', 'american_bulldog_12' -> 'american_bulldog'
    return re.sub(r'_\d+$', '', name)


def read_trainval(path):
    entries = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            name, class_id, species, breed_id = line.split()
            entries.append((name, int(class_id), int(species)))
    return entries


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find('size')
    xml_wh = (int(size.find('width').text), int(size.find('height').text))
    objs = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        xyxy = [float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        objs.append((obj.find('name').text, xyxy))
    return xml_wh, objs


def allocate(counts, frac, total, rng):
    """Largest-remainder allocation of `total` items across strata, proportional to counts."""
    exact = {k: n * frac for k, n in counts.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    keys = sorted(counts, key=lambda k: (-(exact[k] - alloc[k]), rng.random()))
    for k in keys[:total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified_split(names_by_class, seed, val_frac=0.1, test_frac=0.1):
    rng = random.Random(seed)
    counts = {c: len(v) for c, v in names_by_class.items()}
    n = sum(counts.values())
    n_val, n_test = round(n * val_frac), round(n * test_frac)
    val_alloc = allocate(counts, val_frac, n_val, rng)
    test_alloc = allocate(counts, test_frac, n_test, rng)

    splits = {'train': [], 'val': [], 'test': []}
    for c in sorted(names_by_class):
        names = sorted(names_by_class[c])
        rng.shuffle(names)
        v, t = val_alloc[c], test_alloc[c]
        splits['val'] += names[:v]
        splits['test'] += names[v:v + t]
        splits['train'] += names[v + t:]
    return {k: sorted(v) for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/oxford_pets')
    parser.add_argument('--out', default=None, help='default: <root>/coco')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.root, 'coco')
    os.makedirs(out_dir, exist_ok=True)

    entries = read_trainval(os.path.join(args.root, 'annotations', 'trainval.txt'))
    xml_dir = os.path.join(args.root, 'annotations', 'xmls')
    img_dir = os.path.join(args.root, 'images')

    # class id -> breed name, checked for consistency against the file names
    class_names = {}
    for name, cid, _ in entries:
        breed = breed_from_name(name)
        assert class_names.setdefault(cid, breed) == breed, f'class {cid}: {class_names[cid]} vs {breed}'
    assert sorted(class_names) == list(range(1, 38)), sorted(class_names)

    records, skipped = {}, []
    species_mismatch, size_mismatch = 0, []
    for name, cid, species in entries:
        xml_path = os.path.join(xml_dir, name + '.xml')
        if not os.path.exists(xml_path):
            skipped.append(name)
            continue
        xml_wh, objs = parse_xml(xml_path)
        with Image.open(os.path.join(img_dir, name + '.jpg')) as im:
            w, h = im.size
        if (w, h) != xml_wh:
            size_mismatch.append((name, xml_wh, (w, h)))
        boxes = []
        for xml_label, (x1, y1, x2, y2) in objs:
            # species is only a sanity check; the label is always the breed
            if xml_label != ('cat' if species == 1 else 'dog'):
                species_mismatch += 1
            # VOC boxes are 1-based inclusive -> 0-based COCO xywh (same convention as detectron2)
            x1, y1 = max(x1 - 1, 0.), max(y1 - 1, 0.)
            x2, y2 = min(x2, w), min(y2, h)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        records[name] = dict(class_id=cid, width=w, height=h, boxes=boxes)

    names_by_class = defaultdict(list)
    for name, r in records.items():
        names_by_class[r['class_id']].append(name)
    splits = stratified_split(names_by_class, args.seed)

    categories = [{'id': cid - 1, 'name': class_names[cid], 'supercategory': 'pet'} for cid in sorted(class_names)]
    image_ids = {name: i + 1 for i, name in enumerate(sorted(records))}

    stats, ann_id = {}, 1
    all_areas = []
    for split, names in splits.items():
        images, anns = [], []
        for name in names:
            r = records[name]
            images.append({'id': image_ids[name], 'file_name': name + '.jpg', 'width': r['width'], 'height': r['height']})
            for x, y, bw, bh in r['boxes']:
                anns.append({'id': ann_id, 'image_id': image_ids[name], 'category_id': r['class_id'] - 1,
                             'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
                ann_id += 1
        with open(os.path.join(out_dir, f'{split}.json'), 'w') as f:
            json.dump({'images': images, 'annotations': anns, 'categories': categories}, f)
        areas = [a['area'] for a in anns]
        all_areas += areas
        per_img = Counter(a['image_id'] for a in anns)
        stats[split] = dict(images=len(images), boxes=len(anns),
                            multi=sum(1 for v in per_img.values() if v > 1),
                            classes=len({a['category_id'] for a in anns}),
                            small=sum(a < 32 ** 2 for a in areas),
                            medium=sum(32 ** 2 <= a <= 96 ** 2 for a in areas),
                            large=sum(a > 96 ** 2 for a in areas))

    with open(os.path.join(out_dir, 'split_manifest.txt'), 'w') as f:
        f.write(f'# Oxford-IIIT Pet trainval -> 80/10/10 stratified by breed, seed={args.seed}\n')
        for split, names in splits.items():
            f.write(f'[{split}] {len(names)}\n')
            f.writelines(n + '.jpg\n' for n in names)

    multi_names = [n for n, r in records.items() if len(r['boxes']) > 1]
    print(f'trainval.txt entries: {len(entries)}')
    print(f'skipped (no XML): {len(skipped)}  {skipped}')
    print(f'images kept: {len(records)}')
    print(f'XML species tag != trainval species: {species_mismatch}')
    print(f'XML size != actual image size: {len(size_mismatch)}  {size_mismatch[:5]}')
    print(f'number of classes: {len(categories)} (category_id {categories[0]["id"]}..{categories[-1]["id"]})')
    print(f'images with >1 box: {len(multi_names)}  {multi_names}')
    print()
    print(f'{"split":<6} {"images":>7} {"boxes":>7} {">1 box":>7} {"classes":>8} {"<32²":>6} {"32²-96²":>8} {">96²":>6}')
    for split, s in stats.items():
        print(f'{split:<6} {s["images"]:>7} {s["boxes"]:>7} {s["multi"]:>7} {s["classes"]:>8} '
              f'{s["small"]:>6} {s["medium"]:>8} {s["large"]:>6}')
    tot = {k: sum(s[k] for s in stats.values()) for k in ('images', 'boxes', 'multi', 'small', 'medium', 'large')}
    print(f'{"total":<6} {tot["images"]:>7} {tot["boxes"]:>7} {tot["multi"]:>7} {len(categories):>8} '
          f'{tot["small"]:>6} {tot["medium"]:>8} {tot["large"]:>6}')
    print()
    print(f'box area: min {min(all_areas):.0f} px² (~{min(all_areas) ** 0.5:.0f}²), '
          f'max {max(all_areas):.0f} px² (~{max(all_areas) ** 0.5:.0f}²)')
    per_class_split = {sp: Counter(records[n]['class_id'] for n in names) for sp, names in splits.items()}
    for sp, c in per_class_split.items():
        print(f'{sp} images per breed: min {min(c.values())}, max {max(c.values())}')
    print(f'wrote {out_dir}/{{train,val,test}}.json and split_manifest.txt')


if __name__ == '__main__':
    main()
