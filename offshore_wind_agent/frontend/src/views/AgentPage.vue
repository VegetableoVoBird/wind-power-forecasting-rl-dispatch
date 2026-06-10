<script setup>
import { computed, onMounted, ref } from 'vue'
import { useDashboard } from '../composables/useDashboard'

const {
  agentBackend,
  activeStation,
  exportForecastUrl,
  exportReportUrl,
  answer,
  answerMeta,
  asking,
  ensureDashboardLoaded,
  submitQuestion,
  refreshOllamaStatus,
  fmt,
} = useDashboard()

const question = ref('')
const exportToast = ref('')
const refreshingOllama = ref(false)

async function handleRefreshOllama() {
  refreshingOllama.value = true
  try {
    await refreshOllamaStatus()
  } finally {
    refreshingOllama.value = false
  }
}

const quickQuestions = computed(() => [
  '请解释当前系统在验证集上的预测精度如何',
  '请概括一下当前系统的核心分析能力',
  'Q-Learning 相比激进并网策略提升了什么',
  '请用答辩风格解释遗传算法在系统中的作用',
])

function useQuickQuestion(text) {
  question.value = text
}

async function handleAsk() {
  await submitQuestion(question.value)
}

function notifyExport(type) {
  exportToast.value = type === 'forecast' ? '已触发验证结果导出。' : '已触发实验报告导出。'
  setTimeout(() => {
    exportToast.value = ''
  }, 2400)
}

onMounted(() => {
  ensureDashboardLoaded()
})
</script>

<template>
  <div class="page-grid">
    <section class="dual-grid">
      <article class="page-panel agent-hero-card">
        <div class="panel-head">
          <div>
            <p class="eyebrow">智能体中枢</p>
            <h3>智能体中心</h3>
            <p>快问、上下文、导出和回答都集中在一个专业面板里，方便答辩时快速调取信息。</p>
          </div>
        </div>
        <div class="agent-hero-grid">
          <div class="agent-orbit">
            <div class="agent-core"></div>
            <div class="agent-core pulse"></div>
            <div class="agent-node node-a"></div>
            <div class="agent-node node-b"></div>
            <div class="agent-node node-c"></div>
          </div>
          <div class="agent-status-stack">
            <div class="context-card">
              <span>问答后端</span>
              <strong>
                <span v-if="agentBackend?.available" style="color:#4caf50">Ollama 已连接</span>
                <span v-else style="color:#ff9800">规则问答 (Ollama 未连接)</span>
              </strong>
              <small v-if="agentBackend?.model">模型: {{ agentBackend.model }}</small>
              <small v-else>{{ agentBackend?.message || '请确保 Ollama 已启动并点击重连' }}</small>
              <button
                class="button ghost"
                style="margin-top:8px;font-size:12px;padding:4px 12px"
                :disabled="refreshingOllama"
                @click="handleRefreshOllama"
              >
                {{ refreshingOllama ? '重连中...' : '重连 Ollama' }}
              </button>
            </div>
            <div class="context-card">
              <span>当前站点</span>
              <strong>{{ activeStation?.site_name || '--' }}</strong>
              <small>{{ activeStation ? `${fmt(activeStation.avg_risk, 3)} 风险 · ${activeStation.dominant_action}` : '等待站点上下文' }}</small>
            </div>
          </div>
        </div>
      </article>

      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">快捷操作</p>
            <h3>可操作功能</h3>
            <p>快捷提问、导出数据和上下文提示。</p>
          </div>
        </div>
        <div class="quick-question-list">
          <button v-for="item in quickQuestions" :key="item" type="button" class="quick-question" @click="useQuickQuestion(item)">
            {{ item }}
          </button>
        </div>

        <div class="export-row">
          <a class="button primary" :href="exportForecastUrl" @click="notifyExport('forecast')">导出验证结果</a>
          <a class="button ghost" :href="exportReportUrl" @click="notifyExport('report')">导出实验报告</a>
        </div>
        <div v-if="exportToast" class="toast-banner">{{ exportToast }}</div>
      </article>
    </section>

    <section class="dual-grid">
      <article class="page-panel qa-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">问答工作台</p>
            <h3>问答工作台</h3>
            <p>专门处理智能问答，支持精度验证、模型对比、调度策略等各类问题。</p>
          </div>
        </div>
        <div class="qa-box">
          <textarea
            v-model="question"
            placeholder="例如：请解释当前站点为什么采用保守预留而不是激进并网"
          ></textarea>
          <button class="button primary wide" :disabled="asking" @click="handleAsk">
            {{ asking ? '正在回答...' : '向智能体提问' }}
          </button>
          <div class="answer-meta">{{ answerMeta }}</div>
          <div class="answer-box">{{ answer }}</div>
        </div>
      </article>

      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">提问指引</p>
            <h3>提问指引</h3>
            <p>适合答辩和演示的提问方式。</p>
          </div>
        </div>
        <div class="summary-stack">
          <article class="summary-line-card">
            <span>精度验证</span>
            <strong>让智能体说明模型在验证集上的预测精度和误差分布</strong>
            <small>适合展示系统分析能力的可靠性。</small>
          </article>
          <article class="summary-line-card">
            <span>实验答辩</span>
            <strong>让智能体解释 GA、启发式搜索和 RL 的差异</strong>
            <small>适合课程实验展示和答辩总结。</small>
          </article>
          <article class="summary-line-card">
            <span>策略复盘</span>
            <strong>让智能体概括为什么高风险时更保守</strong>
            <small>适合讲调度逻辑与安全收益平衡。</small>
          </article>
        </div>
      </article>
    </section>
  </div>
</template>
