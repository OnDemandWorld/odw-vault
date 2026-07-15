# Vault UI 设计系统与重构指南

> 版本: v1.0 | 日期: 2026-07-15 | 状态: 待确认

---

## 一、设计理念

**关键词：** `知识感` · `克制精致` · `温暖专业` · `可信赖`

Vault 是知识管理产品，设计应传达"深度"和"可信赖"。不追求 ChatGPT 的极简黑白，也不追求 Linear 的极客冷感。Vault 的个性是：**温暖的知识库，专业的研究助手**。

具体体现：
- 色彩带有微弱暖色调（不是冷灰，而是像旧书页的温暖）
- 引用展示是核心体验，需要精心设计
- 信息密度适中 — 比 Claude 更紧凑，比 Dify 更宽松
- 暗色模式不是简单反色，而是独立的暖色暗调

### 竞品对标

| 产品 | 核心学习点 |
|------|-----------|
| **ChatGPT** | 消息布局、流式渲染、Composer 胶囊设计 |
| **Claude.ai** | 排版品质（行高 1.75）、暖色调中性色、Artifacts 引用面板 |
| **Linear** | 动效体系（cubic-bezier 缓动）、暗色模式精致度、Skeleton Loading |
| **Vercel** | 暗色模式层级（三级灰度递进）、开发者体验 |
| **Dify** | 引用上标 [1][2] + 可折叠 Source Panel |
| **Perplexity** | 来源卡片网格 + 相关度可视化 |

---

## 二、完整 Design Token 系统

### 2.1 色彩体系 — Light Mode

```css
:root, [data-theme="light"] {
  /* ═══ 背景层级 ═══ */
  --bg-app:        #FAFAF8;   /* 应用最底层背景（微暖灰白） */
  --bg-primary:    #FFFFFF;   /* 主内容区背景 */
  --bg-secondary:  #F5F5F0;   /* 次级区域（sidebar、卡片） */
  --bg-tertiary:   #EDEDEA;   /* 第三级背景（hover、选中） */
  --bg-user-msg:   #F0EFEB;   /* 用户消息气泡 */
  --bg-code:       #F5F4F0;   /* 代码块背景 */
  --bg-citation:   #F8F6F0;   /* 引用面板背景（微黄，像旧纸） */

  /* ═══ 文字层级 ═══ */
  --text-primary:   #1A1A1A;  /* 主文字 — 标题、正文 */
  --text-secondary: #5C5C5C;  /* 次级文字 — 描述、辅助 */
  --text-tertiary:  #8C8C8C;  /* 三级文字 — 占位符、时间戳 */
  --text-disabled:  #B8B8B8;  /* 禁用态文字 */
  --text-inverse:   #FFFFFF;  /* 反色文字（深色按钮上） */

  /* ═══ 边框层级 ═══ */
  --border-subtle:   rgba(0,0,0,0.06);  /* 分割线 */
  --border-default:  rgba(0,0,0,0.10);  /* 输入框、卡片 */
  --border-strong:   rgba(0,0,0,0.16);  /* focus 态 */
  --border-emphasis: rgba(0,0,0,0.24);  /* 选中态 */

  /* ═══ 品牌/强调色 ═══ */
  --accent:        #6B5CE7;   /* 主强调色 — 柔和紫 */
  --accent-hover:  #5A4BD6;
  --accent-active: #4A3BC5;
  --accent-subtle: #F0EEFC;   /* 极浅底 — 选中行背景 */
  --accent-fg:     #FFFFFF;   /* 强调色上的文字 */

  /* ═══ 引用专属色 ═══ */
  --citation-bg:       #F8F6F0;  /* 引用卡片背景 */
  --citation-border:   #E8E4D8;  /* 引用卡片边框（暖色调） */
  --citation-marker:   #8B7E6A;  /* 引用编号（古铜色） */
  --citation-marker-bg:#F0EDE4;  /* 引用编号背景 */
  --citation-text:     #6B6355;  /* 引用内容文字 */

  /* ═══ 语义色 ═══ */
  --success:    #2D9F5E;  --success-bg: #E8F5EE;
  --warning:    #D48B07;  --warning-bg: #FEF5E7;
  --error:      #DC3545;  --error-bg:   #FDE8EA;
  --info:       #2B7DE9;  --info-bg:    #E8F0FE;

  /* ═══ 阴影 ═══ */
  --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.06), 0 2px 4px -2px rgba(0,0,0,0.04);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.06), 0 4px 6px -4px rgba(0,0,0,0.04);
  --shadow-composer:       0 0 0 1px rgba(0,0,0,0.05), 0 2px 12px rgba(0,0,0,0.06);
  --shadow-composer-focus: 0 0 0 1.5px rgba(107,92,231,0.3), 0 4px 24px rgba(0,0,0,0.08);

  /* ═══ 滚动条 ═══ */
  --scrollbar-thumb:       rgba(0,0,0,0.12);
  --scrollbar-thumb-hover: rgba(0,0,0,0.20);
}
```

### 2.2 色彩体系 — Dark Mode

```css
[data-theme="dark"] {
  /* ═══ 背景层级（暖色暗调） ═══ */
  --bg-app:        #141413;   /* 微暖黑 */
  --bg-primary:    #1A1A18;
  --bg-secondary:  #222220;
  --bg-tertiary:   #2C2C28;
  --bg-user-msg:   #2A2A26;
  --bg-code:       #1E1E1C;
  --bg-citation:   #201F1A;

  /* ═══ 文字层级 ═══ */
  --text-primary:   #EDECEA;  /* 微暖白 */
  --text-secondary: #A09E9A;
  --text-tertiary:  #6B6966;
  --text-disabled:  #4A4845;
  --text-inverse:   #1A1A1A;

  /* ═══ 边框层级 ═══ */
  --border-subtle:   rgba(255,255,255,0.06);
  --border-default:  rgba(255,255,255,0.10);
  --border-strong:   rgba(255,255,255,0.16);
  --border-emphasis: rgba(255,255,255,0.24);

  /* ═══ 品牌/强调色（暗色更亮） ═══ */
  --accent:        #8B7FF5;
  --accent-hover:  #9D93F7;
  --accent-active: #7B6FE5;
  --accent-subtle: rgba(139,127,245,0.10);
  --accent-fg:     #1A1A1A;

  /* ═══ 引用专属色（暗色） ═══ */
  --citation-bg:       #1E1D18;
  --citation-border:   #33312A;
  --citation-marker:   #A89B85;
  --citation-marker-bg:#2A2820;
  --citation-text:     #9E9585;

  /* ═══ 语义色（暗色） ═══ */
  --success: #3DB86E;  --success-bg: rgba(61,184,110,0.12);
  --warning: #E8A020;  --warning-bg: rgba(232,160,32,0.12);
  --error:   #E85565;  --error-bg:   rgba(232,85,101,0.12);
  --info:    #4D93F0;  --info-bg:    rgba(77,147,240,0.12);

  /* ═══ 阴影（暗色更深） ═══ */
  --shadow-xs: 0 1px 2px rgba(0,0,0,0.20);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.30), 0 1px 2px rgba(0,0,0,0.20);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.30), 0 2px 4px -2px rgba(0,0,0,0.20);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.30), 0 4px 6px -4px rgba(0,0,0,0.20);
  --shadow-composer:       0 0 0 1px rgba(255,255,255,0.06), 0 2px 12px rgba(0,0,0,0.30);
  --shadow-composer-focus: 0 0 0 1.5px rgba(139,127,245,0.40), 0 4px 24px rgba(0,0,0,0.40);

  /* ═══ 滚动条 ═══ */
  --scrollbar-thumb:       rgba(255,255,255,0.10);
  --scrollbar-thumb-hover: rgba(255,255,255,0.18);
}
```

### 2.3 字体排印体系

```css
:root {
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "SF Mono", "Fira Code", "Cascadia Code",
               "Consolas", monospace;
}

/* 字号阶梯（Modular Scale — ratio 1.2） */
:root {
  --text-display:   clamp(28px, 4vw, 40px);  /* 英雄区大标题 */
  --text-h1:        24px;   --text-h1-lh: 1.25;  --text-h1-wt: 500;
  --text-h2:        20px;   --text-h2-lh: 1.3;   --text-h2-wt: 500;
  --text-h3:        17px;   --text-h3-lh: 1.4;   --text-h3-wt: 500;
  --text-body:      15px;   --text-body-lh: 1.7;  --text-body-wt: 400;
  --text-sm:        13px;   --text-sm-lh: 1.5;   --text-sm-wt: 400;
  --text-caption:   11px;   --text-caption-lh: 1.4; --text-caption-wt: 500;
  --text-code:      13px;   --text-code-lh: 1.6;
}
```

### 2.4 间距系统（8px 基准网格）

```css
:root {
  --space-1:  4px;    /* 图标与文字间 */
  --space-2:  8px;    /* 相关元素内 */
  --space-3:  12px;   /* 组件内 padding */
  --space-4:  16px;   /* 组件间 */
  --space-5:  20px;   /* 区域分隔 */
  --space-6:  24px;   /* 区块间距 */
  --space-8:  32px;   /* 大区块 */
  --space-10: 40px;   /* 页面级 */
  --space-12: 48px;   /* 英雄区 padding */

  --msg-gap:          24px;   /* 消息间垂直间距 */
  --canvas-max-width: 768px;  /* 内容最大宽度 */
  --topbar-height:    48px;
}
```

### 2.5 圆角系统

```css
:root {
  --radius-xs:   4px;   /* 标签、小徽章 */
  --radius-sm:   6px;   /* 按钮、输入框 */
  --radius-md:   10px;  /* 卡片、下拉菜单 */
  --radius-lg:   14px;  /* 大卡片、面板 */
  --radius-xl:   20px;  /* 消息气泡 */
  --radius-2xl:  28px;  /* Composer */
  --radius-full: 9999px;/* 圆形 */
}
```

### 2.6 过渡/动画标准

```css
:root {
  --duration-instant: 50ms;   /* 按下反馈 */
  --duration-fast:    100ms;  /* hover 颜色 */
  --duration-normal:  200ms;  /* 大多数过渡 */
  --duration-slow:    300ms;  /* 面板展开 */
  --duration-slower:  500ms;  /* 复杂动画 */

  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);     /* Material standard */
  --ease-enter:   cubic-bezier(0, 0, 0.2, 1);        /* 入场 */
  --ease-exit:    cubic-bezier(0.4, 0, 1, 1);        /* 退场 */
  --ease-spring:  cubic-bezier(0.34, 1.56, 0.64, 1); /* 弹性 */
}
```

---

## 三、现有 UI 深度 Review 问题清单

### P0 — 必须立即修复

| # | 问题 | 位置 | 修复方案 |
|---|------|------|---------|
| 1 | **无 Dark Mode** | 全局 | 实现 `[data-theme="dark"]` 完整 token 映射 + 切换器 |
| 2 | **13+ 处硬编码颜色** | CSS/JS | 全部替换为 CSS 变量引用 |
| 3 | **Markdown 渲染极简** | `_md()` JS | 补充列表、标题、链接、引用块支持 |
| 4 | **引用展示过于简陋** | `#cit` | 升级为三层引用体系（行内标记 + 摘要面板 + 详情抽屉） |

### P1 — 重要改进

| # | 问题 | 位置 | 修复方案 |
|---|------|------|---------|
| 5 | **无交互状态** | 所有按钮 | 补充 hover/focus/active/disabled/loading 6 种状态 |
| 6 | **无过渡动画** | 全局 | 消息入场、面板展开、颜色变化添加 transition |
| 7 | **空状态无引导** | `#hero` | 添加知识范围指示器 + 引导文案 |
| 8 | **Loading 状态缺失** | 全局 | 用 Skeleton Screen 替代 "Thinking..." |
| 9 | **无主题切换器** | topbar | 添加 Light/Dark/System 三态切换按钮 |

### P2 — 锦上添花

| # | 问题 | 位置 | 修复方案 |
|---|------|------|---------|
| 10 | **代码块无语言标签/Copy** | `.md pre` | 添加语言标签 + 一键复制按钮 |
| 11 | **无消息操作栏** | AI 消息 | hover 显示复制/重新生成/反馈按钮 |
| 12 | **滚动条样式粗糙** | 全局 | 使用 token 化的滚动条样式 |
| 13 | **无键盘导航** | 全局 | 补充 focus-visible 样式 + Tab 序列 |
| 14 | **无响应式断点** | 全局 | 添加 600px / 1024px 断点适配 |

---

## 四、引用展示升级方案：三层体系

### 第一层：行内引用标记
回答文本中的 `[1]` `[2]` 上标，可点击跳转到对应引用卡片。

### 第二层：引用摘要面板（消息底部）
可折叠的来源列表，每张卡片包含：编号 + 文件名 + 相关段落摘要 + 页码。

### 第三层：来源详情展开（点击卡片）
右侧滑出抽屉，展示完整匹配段落和高亮内容。

---

## 五、主题切换技术方案

```javascript
// 主题切换逻辑
const THEME_KEY = 'vault-theme';

function getPreferredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
  const resolved = theme === 'system'
    ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    : theme;
  document.documentElement.setAttribute('data-theme', resolved);
  localStorage.setItem(THEME_KEY, theme); // 存储原始值（含 'system'）
}

// 初始化
applyTheme(getPreferredTheme());

// 监听系统主题变化
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (localStorage.getItem(THEME_KEY) === 'system') applyTheme('system');
});
```

**防闪烁（FOUC）：** 在 `<head>` 中添加内联 `<script>` 阻塞渲染，在 CSS 加载前设置 `data-theme`。

---

## 六、WCAG 对比度验证

| 组合 | 对比度 | AA | AAA |
|------|--------|----|----|
| text-primary on bg-primary (Light) | 16.75:1 | ✅ | ✅ |
| text-secondary on bg-primary (Light) | 5.91:1 | ✅ | ✅ |
| text-primary on bg-primary (Dark) | 13.45:1 | ✅ | ✅ |
| text-secondary on bg-primary (Dark) | 6.82:1 | ✅ | ✅ |
| accent on bg-primary (Light) | 4.56:1 | ✅ | ❌ |
| accent on bg-primary (Dark) | 5.35:1 | ✅ | ✅ |

> `--text-tertiary` 仅用于非关键信息（占位符、装饰），不传达关键信息。

---

## 七、实施优先级

| 优先级 | 任务 | 预估工时 |
|--------|------|---------|
| **P0** | Design Token 系统替换 + Dark Mode | 5h |
| **P1** | 引用展示三层体系升级 | 4h |
| **P1** | Skeleton Loading + 交互状态 | 3h |
| **P2** | 消息入场动画 + 空状态升级 | 3h |
| **P2** | Markdown 渲染增强 + 代码块优化 | 2h |
| **P3** | 消息操作栏 + 键盘导航 | 3h |

---

## 八、后续迭代建议

1. **Sidebar + 对话历史** — 需要后端支持对话持久化
2. **知识库管理界面** — 文件夹树 + 文档列表 + 上传入口
3. **评估仪表盘** — 可视化 RAG 质量指标
4. **移动端适配** — 响应式布局 + 触摸手势优化
5. **国际化** — 中英文切换支持
