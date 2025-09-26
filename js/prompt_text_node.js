import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Ken-Chen_VideoUtilities.PromptTextNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "Prompt_Text_Node") {
            // 当前节点为纯文本输出，无需额外前端逻辑。
            // 保留扩展以确保前端可扩展性和与参考项目一致的注册结构。
        }
    },
});


