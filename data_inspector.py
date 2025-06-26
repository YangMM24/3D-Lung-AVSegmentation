import numpy as np
from pathlib import Path

def inspect_npz_file(file_path):
    """
    检查单个NPZ文件的内容
    """
    try:
        data = np.load(file_path)
        print(f"\n📁 {file_path.name}")
        print(f"   包含数组: {list(data.keys())}")
        
        for key in data.keys():
            arr = data[key]
            print(f"   {key}:")
            print(f"     - 形状: {arr.shape}")
            print(f"     - 数据类型: {arr.dtype}")
            print(f"     - 数值范围: [{arr.min():.2f}, {arr.max():.2f}]")
            
            # 如果是布尔或整数数组，显示唯一值
            if arr.dtype == np.bool_ or (np.issubdtype(arr.dtype, np.integer) and arr.max() < 10):
                unique_vals = np.unique(arr)
                if len(unique_vals) <= 10:
                    print(f"     - 唯一值: {unique_vals}")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"❌ 无法读取 {file_path.name}: {str(e)}")
        return False

def quick_inspect():
    """
    快速检查前5个样本的数据结构
    """
    base_path = Path("D:/yangmiaomiao/study/ucl/final/3D_lung_segmentation/dataset")
    
    print("🔍 检查NPZ文件数据结构")
    print("=" * 60)
    
    # 检查样本001的所有文件
    sample_id = "001"
    
    print(f"📊 样本 {sample_id} 详细信息:")
    
    # CT扫描
    ct_file = base_path / "ct_scan" / f"{sample_id}.npz"
    if ct_file.exists():
        print("\n🫁 CT扫描数据:")
        inspect_npz_file(ct_file)
    
    # 动脉标注
    artery_file = base_path / "annotation" / "artery" / f"{sample_id}.npz"
    if artery_file.exists():
        print("\n🔴 动脉标注数据:")
        inspect_npz_file(artery_file)
    
    # 静脉标注
    vein_file = base_path / "annotation" / "vein" / f"{sample_id}.npz"
    if vein_file.exists():
        print("\n🔵 静脉标注数据:")
        inspect_npz_file(vein_file)
    
    # 检查其他样本是否存在
    print(f"\n📋 检查样本002-005是否存在:")
    for i in range(2, 6):
        sample_id = f"{i:03d}"
        ct_exists = (base_path / "ct_scan" / f"{sample_id}.npz").exists()
        artery_exists = (base_path / "annotation" / "artery" / f"{sample_id}.npz").exists()
        vein_exists = (base_path / "annotation" / "vein" / f"{sample_id}.npz").exists()
        
        status = "✓" if all([ct_exists, artery_exists, vein_exists]) else "⚠️"
        print(f"   {status} 样本{sample_id}: CT={ct_exists}, 动脉={artery_exists}, 静脉={vein_exists}")

if __name__ == "__main__":
    quick_inspect()