import os

# 快速检查你的数据集结构
data_dir = 'D:/yangmiaomiao/study/ucl/final/3D_lung_segmentation/dataset'

print("=== 快速数据集检查 ===")
print(f"数据集路径: {data_dir}")

# 显示主目录内容
print(f"\n主目录内容:")
if os.path.exists(data_dir):
    items = os.listdir(data_dir)
    for item in items:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")

# 检查是否有annotation文件夹
annotation_path = os.path.join(data_dir, 'annotation')
print(f"\nannotation文件夹存在: {os.path.exists(annotation_path)}")

if os.path.exists(annotation_path):
    print(f"\nannotation文件夹内容:")
    for item in os.listdir(annotation_path):
        print(f"  📁 {item}/")
        
    # 检查artery和vein子文件夹
    artery_path = os.path.join(annotation_path, 'artery')
    vein_path = os.path.join(annotation_path, 'vein')
    
    if os.path.exists(artery_path):
        artery_files = len([f for f in os.listdir(artery_path) if f.endswith('.npz')])
        print(f"  artery文件夹: {artery_files} 个npz文件")
    
    if os.path.exists(vein_path):
        vein_files = len([f for f in os.listdir(vein_path) if f.endswith('.npz')])
        print(f"  vein文件夹: {vein_files} 个npz文件")

# 检查ct_scan文件夹
ct_scan_path = os.path.join(data_dir, 'ct_scan')
if os.path.exists(ct_scan_path):
    ct_files = len([f for f in os.listdir(ct_scan_path) if f.endswith('.npz')])
    print(f"\nct_scan文件夹: {ct_files} 个npz文件")