from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display

def load_step_file(filename):
    reader = STEPControl_Reader()
    status = reader.ReadFile(filename)

    if status != IFSelect_RetDone:
        print("❌ 读取 STEP 文件失败:", filename)
        return

    # 加载到模型中
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape

def main():
    filename = "/home/ein/Dev/OcctStuff/box.stp"  # 修改为你的 STEP 文件路径
    shape = load_step_file(filename)
    if not shape:
        return

    # 初始化可视化窗口
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    start_display()

if __name__ == "__main__":
    main()