import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Ken-Chen_VideoUtilities.RGBEmptyImage",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== "RGB_Empty_Image") return;
    // 该节点为静态图像输出，无需额外前端逻辑；保留扩展占位与兼容参考项目结构。
  }
});


