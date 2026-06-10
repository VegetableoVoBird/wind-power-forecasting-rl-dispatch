<script setup>
import { computed, reactive, ref, watch } from 'vue'

const props = defineProps({
  series: { type: Array, default: () => [] },
  fmt: { type: Function, required: true },
  viewMode: { type: String, default: 'compare' },
  validationMode: { type: Boolean, default: false },
})

const lineVisible = reactive({
  actual: true,
  predicted: true,
  dispatch: true,
  risk: false,
})

function toggleLine(key) { lineVisible[key] = !lineVisible[key] }

// ---- 时间窗口 ----
const windowStart = ref(0)
const windowSize = ref(192)
const maxStart = computed(() => Math.max(0, props.series.length - windowSize.value))

watch(() => props.series.length, () => { windowStart.value = 0 })

const visibleSeries = computed(() => {
  const raw = props.series.slice(windowStart.value, windowStart.value + windowSize.value)
  if (raw.length <= 600) return raw
  const step = Math.ceil(raw.length / 600)
  return raw.filter((_, i) => i % step === 0)
})

const timeRangeLabel = computed(() => {
  if (!visibleSeries.value.length) return ''
  const f = visibleSeries.value[0]?.timestamp || ''
  const l = visibleSeries.value[visibleSeries.value.length - 1]?.timestamp || ''
  return `${f} ~ ${l}`
})

const totalRangeLabel = computed(() => {
  if (!props.series.length) return ''
  const f = props.series[0]?.timestamp || ''
  const l = props.series[props.series.length - 1]?.timestamp || ''
  return `${f} ~ ${l}  (共${props.series.length}个时间片)`
})

// ---- SVG 渲染 ----
function buildSvg() {
  const vs = visibleSeries.value
  if (!vs.length) return ''
  const W = 920; const H = 280; const P = 30

  const hasActual = props.validationMode && vs.some(r => Number(r.actual_power_mw) > 0)
  const predicted = vs.map(r => Number(r.predicted_power_mw) || 0)
  const actual = hasActual ? vs.map(r => Number(r.actual_power_mw) || 0) : []
  const dispatch = vs.map(r => Number(r.dispatch_power_mw) || 0)
  const risk = vs.map(r => { const v = Number(r.risk_score); return Number.isFinite(v) ? v : 0 })

  const useRiskScale = props.viewMode === 'risk'
  const useDispatchFocus = props.viewMode === 'dispatch'
  const allVals = [...predicted, ...dispatch, ...actual]
  const maxV = useRiskScale
    ? Math.max(...risk, 0.8) * 1.08   // 风险模式: 至少0.8, 顶部留8%padding
    : Math.max(...allVals.filter(v => Number.isFinite(v)), 1e-6) * 1.03

  function toPath(vals) {
    if (!vals.length) return ''
    return vals.map((v, i) => {
      const x = P + (i / Math.max(vals.length - 1, 1)) * (W - P * 2)
      const y = H - P - (v / maxV) * (H - P * 2)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
  }

  // 网格线 + Y轴标签
  const gridCount = useRiskScale ? 6 : 5
  const grid = Array.from({ length: gridCount }, (_, i) => {
    const y = P + ((H - P * 2) / (gridCount - 1)) * i
    const v = maxV - (maxV / (gridCount - 1)) * i
    const label = useRiskScale ? props.fmt(v, 2) : props.fmt(v, 0) + ' MW'
    return `<line x1="${P}" y1="${y.toFixed(1)}" x2="${W-P}" y2="${y.toFixed(1)}" class="gridline"/>
      <text x="8" y="${(y+4).toFixed(1)}" class="axis-label">${label}</text>`
  }).join('')

  // X轴时间标签
  const labels = [0, Math.floor(vs.length * 0.25), Math.floor(vs.length * 0.5), Math.floor(vs.length * 0.75), vs.length - 1]
    .map(i => {
      const pt = vs[Math.min(i, vs.length - 1)]
      const x = P + (i / Math.max(vs.length - 1, 1)) * (W - P * 2)
      return `<text x="${x.toFixed(1)}" y="${H-8}" text-anchor="middle" class="axis-label">${pt?.timestamp || ''}</text>`
    }).join('')

  let elements = grid

  if (useRiskScale) {
    // ---- 风险模式: 面积填充 + 参考线 ----
    // 面积填充 (渐变: 顶部透明, 底部半透明)
    const areaPath = toPath(risk)
    const areaBottom = H - P
    const firstX = P
    const lastX = P + (W - P * 2)
    const areaD = `M${firstX.toFixed(1)},${areaBottom.toFixed(1)} L${toPath(risk)} L${lastX.toFixed(1)},${areaBottom.toFixed(1)} Z`
    elements += `\n<path d="${areaD}" class="path-risk-area"/>`

    // 风险线
    elements += `\n<path d="${toPath(risk)}" class="path-risk"/>`

    // 阈值参考线: 0.45 (中风险) 和 0.75 (高风险)
    const thresholds = [
      { val: 0.45, label: '中风险 0.45', cls: 'threshold-medium' },
      { val: 0.75, label: '高风险 0.75', cls: 'threshold-high' },
    ]
    thresholds.forEach(t => {
      const ty = H - P - (t.val / maxV) * (H - P * 2)
      if (ty > P && ty < H - P) {
        elements += `\n<line x1="${P}" y1="${ty.toFixed(1)}" x2="${W-P}" y2="${ty.toFixed(1)}" class="${t.cls}"/>`
        elements += `\n<text x="${W-P-4}" y="${(ty-4).toFixed(1)}" text-anchor="end" class="threshold-label">${t.label}</text>`
      }
    })
  } else if (useDispatchFocus) {
    // ---- 调度聚焦模式: 突出调度线, 其他线半透明 ----
    if (hasActual && lineVisible.actual) elements += `\n<path d="${toPath(actual)}" class="path-actual path-dim"/>`
    if (lineVisible.predicted) elements += `\n<path d="${toPath(predicted)}" class="path-forecast path-dim"/>`
    if (lineVisible.dispatch) elements += `\n<path d="${toPath(dispatch)}" class="path-dispatch only"/>`
  } else {
    // ---- 功率对比模式 ----
    if (hasActual && lineVisible.actual) elements += `\n<path d="${toPath(actual)}" class="path-actual"/>`
    if (lineVisible.predicted) elements += `\n<path d="${toPath(predicted)}" class="path-forecast"/>`
    if (lineVisible.dispatch) elements += `\n<path d="${toPath(dispatch)}" class="path-dispatch"/>`
  }

  elements += `\n${labels}`
  return elements
}

const svgMarkup = computed(() => buildSvg())

// ---- 时间轴拖杆 ----
const isDragging = ref(false)

function scrubberLeft() {
  if (!props.series.length) return 0
  return (windowStart.value / props.series.length) * 100
}
function scrubberWidth() {
  if (!props.series.length) return 100
  return (windowSize.value / props.series.length) * 100
}

function onScrubberMouseDown(e) {
  isDragging.value = true
  document.addEventListener('mousemove', onScrubberMouseMove)
  document.addEventListener('mouseup', onScrubberMouseUp)
}
function onScrubberMouseMove(e) {
  if (!isDragging.value || !props.series.length) return
  const bar = document.querySelector('.scrubber-bar')
  if (!bar) return
  const rect = bar.getBoundingClientRect()
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
  windowStart.value = Math.round(pct * maxStart.value)
  windowStart.value = Math.max(0, Math.min(maxStart.value, windowStart.value))
}
function onScrubberMouseUp() {
  isDragging.value = false
  document.removeEventListener('mousemove', onScrubberMouseMove)
  document.removeEventListener('mouseup', onScrubberMouseUp)
}

function scrollWindow(d) { windowStart.value = Math.max(0, Math.min(maxStart.value, windowStart.value + d)) }
function zoomTo(days) { windowSize.value = days * 96; windowStart.value = Math.min(windowStart.value, maxStart.value) }
</script>

<template>
  <div class="chart-surface">
    <div class="chart-controls">
      <div class="zoom-group">
        <button class="mode-chip" :class="{active:windowSize===96}" @click="zoomTo(1)">1天</button>
        <button class="mode-chip" :class="{active:windowSize===192}" @click="zoomTo(2)">2天</button>
        <button class="mode-chip" :class="{active:windowSize===480}" @click="zoomTo(5)">5天</button>
        <button class="mode-chip" :class="{active:windowSize===960}" @click="zoomTo(10)">10天</button>
        <button class="mode-chip" :class="{active:windowSize>=series.length}" @click="windowSize=series.length;windowStart=0">全部</button>
      </div>
      <div class="scroll-group">
        <button class="mode-chip" :disabled="windowStart===0" @click="scrollWindow(-96)">◀</button>
        <span class="range-label">{{ timeRangeLabel }}</span>
        <button class="mode-chip" :disabled="windowStart>=maxStart" @click="scrollWindow(96)">▶</button>
      </div>
    </div>

    <div v-if="series.length > windowSize" class="scrubber-bar" @mousedown="onScrubberMouseDown">
      <div class="scrubber-track">
        <div class="scrubber-window" :style="{left:scrubberLeft()+'%', width:scrubberWidth()+'%'}"></div>
      </div>
      <div class="scrubber-label">{{ totalRangeLabel }}</div>
    </div>

    <svg viewBox="0 0 920 280" preserveAspectRatio="none" class="power-chart" v-html="svgMarkup"></svg>

    <div class="chart-legend clickable">
      <span v-if="validationMode" @click="toggleLine('actual')" :class="{ dimmed: !lineVisible.actual }">
        <i class="legend-swatch actual"></i>实际功率
      </span>
      <span @click="toggleLine('predicted')" :class="{ dimmed: !lineVisible.predicted }">
        <i class="legend-swatch forecast"></i>模型预测
      </span>
      <span @click="toggleLine('dispatch')" :class="{ dimmed: !lineVisible.dispatch }">
        <i class="legend-swatch dispatch"></i>调度建议
      </span>
    </div>
  </div>
</template>
