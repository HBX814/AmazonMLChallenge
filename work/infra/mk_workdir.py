# mk_workdir.py <src_work> <dst_work>: dst = symlinks to every entry of src except pred/ (fresh), for variant predict runs
import os, sys
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dst, "pred"), exist_ok=True)
for e in os.listdir(src):
    if e == "pred":
        continue
    d = os.path.join(dst, e)
    if not os.path.lexists(d):
        os.symlink(os.path.join(src, e), d)
print(sorted(os.listdir(dst)))
