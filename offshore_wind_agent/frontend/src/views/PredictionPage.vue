<script setup>
import { computed, onMounted, ref } from 'vue'
import { useDashboard } from '../composables/useDashboard'

const API_BASE = '/api'

const {
  stationOverview,
  currentSiteId,
  validationComparison,
  ensureDashboardLoaded,
  ensureSiteLoaded,
  fmt,
} = useDashboard()

// ---- 文件上传预测 ----
const uploadFile = ref(null)
const uploading = ref(false)
const uploadMsg = ref('')
const predictResult = ref(null)  // { columns, rows, total, filename }

function onFileChange(e) {
  uploadFile.value = e.target.files[0] || null
  uploadMsg.value = ''
}

async function handleUpload() {
  if (!uploadFile.value) { uploadMsg.value = '请先选择 CSV 文件'; return }
  uploading.value = true; uploadMsg.value = '正在预测...'; predictResult.value = null
  try {
    const form = new FormData(); form.append('file', uploadFile.value)
    const resp = await fetch(`${API_BASE}/upload/predict`, { method: 'POST', body: form })
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.error || '上传失败') }
    predictResult.value = await resp.json()
    const r = predictResult.value
    if (r.has_actual) {
      uploadMsg.value = `完成, 共 ${r.total} 条 | MAE=${fmt(r.mae_mw)}MW R²=${fmt(r.r2,4)}`
    } else {
      uploadMsg.value = `完成, 共 ${r.total} 条预测结果`
    }
  } catch (e) { uploadMsg.value = `失败: ${e.message}` }
  finally { uploading.value = false }
}

async function downloadResult() {
  if (!uploadFile.value) return
  const form = new FormData(); form.append('file', uploadFile.value)
  const resp = await fetch(`${API_BASE}/upload/predict/download`, { method: 'POST', body: form })
  const blob = await resp.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `predicted_${uploadFile.value.name}`
  a.click(); URL.revokeObjectURL(url)
}

const selectedSiteId = ref('')
const sites = computed(() => stationOverview.value)
const overall = computed(() => validationComparison.value || {})
const overallMae = computed(() => overall.value?.overall_mae_mw || 0)
const overallRmse = computed(() => overall.value?.overall_rmse_mw || 0)
const overallR2 = computed(() => overall.value?.overall_r2 || 0)
const overallMape = computed(() => overall.value?.overall_mape_pct || 0)
const errDist = computed(() => overall.value?.error_distribution || {})
const bySiteData = computed(() => overall.value?.by_site || [])

// 当前选中站点
const currentSiteComp = computed(() =>
  bySiteData.value.find(s => s.site_id === selectedSiteId.value) || null
)

// 取预测vs实际序列数据 (用整体序列或站点序列)
const chartSeries = computed(() => {
  if (currentSiteComp.value?.comparison_series?.length) {
    return currentSiteComp.value.comparison_series
  }
  return overall.value?.overall_comparison_series || []
})

// SVG 路径生成
function buildSvg(series) {
  if (!series.length) return ''
  const W = 920; const H = 320; const P = 30
  const actualVals = series.map(r => Number(r.actual_power_mw) || 0)
  const predVals = series.map(r => Number(r.predicted_power_mw) || 0)
  const maxV = Math.max(...actualVals, ...predVals, 1)

  function path(vals) {
    return vals.map((v, i) => {
      const x = P + (i / Math.max(vals.length - 1, 1)) * (W - P * 2)
      const y = H - P - (v / maxV) * (H - P * 2)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
  }

  const grid = Array.from({ length: 5 }, (_, i) => {
    const y = P + ((H - P * 2) / 4) * i
    const v = maxV - (maxV / 4) * i
    return `<line x1="${P}" y1="${y}" x2="${W-P}" y2="${y}" class="gridline"/>
      <text x="8" y="${y+4}" class="axis-label">${fmt(v,0)} MW</text>`
  }).join('')

  const labels = [0, Math.floor(series.length*0.25), Math.floor(series.length*0.5), Math.floor(series.length*0.75), series.length-1]
    .map(i => {
      const pt = series[Math.min(i, series.length-1)]
      const x = P + (i / Math.max(series.length-1, 1)) * (W - P*2)
      return `<text x="${x}" y="${H-6}" text-anchor="middle" class="axis-label">${pt?.timestamp||''}</text>`
    }).join('')

  return `${grid}
    <path d="${path(actualVals)}" class="path-actual"/>
    <path d="${path(predVals)}" class="path-forecast"/>
    ${labels}`
}

const svgMarkup = computed(() => buildSvg(chartSeries.value))

// 选站点
function selectSite(siteId) {
  selectedSiteId.value = siteId
  ensureSiteLoaded(siteId)
}

onMounted(async () => {
  await ensureDashboardLoaded()
  if (!selectedSiteId.value && sites.value.length) {
    selectedSiteId.value = sites.value[0].site_id
    await ensureSiteLoaded(sites.value[0].site_id)
  }
})
</script>

<template>
  <div class="page-grid">
    <!-- 整体精度 -->
    <section class="top-strip">
      <article class="page-panel metric-tile">
        <span>整体 MAE</span>
        <strong>{{ fmt(overallMae) }} MW</strong>
        <small>验证集平均绝对误差</small>
      </article>
      <article class="page-panel metric-tile">
        <span>整体 RMSE</span>
        <strong>{{ fmt(overallRmse) }} MW</strong>
        <small>验证集均方根误差</small>
      </article>
      <article class="page-panel metric-tile">
        <span>整体 MAPE</span>
        <strong>{{ fmt(overallMape) }}%</strong>
        <small>平均绝对百分比误差</small>
      </article>
      <article class="page-panel metric-tile">
        <span>R² 决定系数</span>
        <strong>{{ fmt(overallR2, 4) }}</strong>
        <small>越接近1拟合越好</small>
      </article>
    </section>

    <!-- 误差分布 -->
    <section v-if="errDist.mean" class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">误差统计</p>
          <h3>验证集预测误差分布</h3>
          <p>均值 {{ fmt(errDist.mean) }} MW · 中位数 {{ fmt(errDist.median) }} MW · P25 {{ fmt(errDist.p25) }} MW · P75 {{ fmt(errDist.p75) }} MW · P90 {{ fmt(errDist.p90) }} MW · P95 {{ fmt(errDist.p95) }} MW · 最大 {{ fmt(errDist.max) }} MW</p>
        </div>
      </div>
      <!-- 误差分布条 -->
      <div class="error-bar-row">
        <div class="error-seg good" :style="{flex: errDist.within_1mw_pct || 0}">
          <small>&lt;1MW<br>{{ errDist.within_1mw_pct }}%</small>
        </div>
        <div class="error-seg ok" :style="{flex: Math.max(0, (errDist.within_3mw_pct||0) - (errDist.within_1mw_pct||0))}">
          <small>1-3MW<br>{{ fmt(Math.max(0, (errDist.within_3mw_pct||0) - (errDist.within_1mw_pct||0)), 1) }}%</small>
        </div>
        <div class="error-seg bad" :style="{flex: Math.max(1, 100 - (errDist.within_3mw_pct||0))}">
          <small>&gt;3MW<br>{{ fmt(Math.max(0, 100 - (errDist.within_3mw_pct||0)), 1) }}%</small>
        </div>
      </div>
    </section>

    <!-- 站点切换 + 预测vs实际 -->
    <section class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">预测对比</p>
          <h3>{{ currentSiteComp ? currentSiteComp.site_id.toUpperCase() + ' — ' : '' }}预测功率 vs 实际功率</h3>
          <p>绿色=实际功率，青色=模型预测。选取验证集中连续时间片进行逐点对比。</p>
        </div>
        <div class="chip-group">
          <button
            v-for="site in sites"
            :key="site.site_id"
            type="button"
            class="mode-chip"
            :class="{ active: selectedSiteId === site.site_id }"
            @click="selectSite(site.site_id)"
          >
            {{ site.site_name }}
          </button>
        </div>
      </div>

      <!-- 当前站点精度 -->
      <div v-if="currentSiteComp" class="site-mae-bar">
        <span>{{ currentSiteComp.site_id.toUpperCase() }}</span>
        <strong>MAE {{ fmt(currentSiteComp.mae_mw) }} MW · R² {{ fmt(currentSiteComp.r2, 4) }}</strong>
        <small>RMSE {{ fmt(currentSiteComp.rmse_mw) }} MW · MAPE {{ fmt(currentSiteComp.mape_pct) }}% · 实际均值 {{ fmt(currentSiteComp.actual_mean_mw) }} MW · 预测均值 {{ fmt(currentSiteComp.pred_mean_mw) }} MW</small>
      </div>

      <div class="chart-surface">
        <svg viewBox="0 0 920 320" preserveAspectRatio="none" class="power-chart" v-html="svgMarkup"></svg>
        <div class="chart-legend">
          <span><i class="legend-swatch actual"></i>实际功率</span>
          <span><i class="legend-swatch forecast"></i>模型预测</span>
        </div>
      </div>
    </section>

    <!-- 动态数据加载 -->
    <section class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">数据加载</p>
          <h3>上传测试集 CSV 获取预测结果</h3>
          <p>上传格式与默认测试集相同的 CSV 文件（含站点编号、时间、气象变量，不含实际功率），系统将自动预测并返回结果。</p>
        </div>
      </div>
      <div class="upload-row">
        <input type="file" accept=".csv" @change="onFileChange" class="filter-input" style="max-width:400px" />
        <button class="button primary" :disabled="uploading" @click="handleUpload">
          {{ uploading ? '预测中...' : '上传并预测' }}
        </button>
        <button v-if="predictResult" class="button ghost" @click="downloadResult">下载 CSV</button>
        <a class="button ghost" :href="`${API_BASE}/download/template`">下载模板</a>
      </div>
      <div v-if="uploadMsg" class="toast-banner" style="margin-top:10px">{{ uploadMsg }}</div>

      <!-- 预测结果表格 -->
      <div v-if="predictResult" style="margin-top:16px; max-height:400px; overflow:auto">
        <table class="result-table">
          <thead>
            <tr><th v-for="col in predictResult.columns" :key="col">{{ col }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in predictResult.rows.slice(0, 100)" :key="i">
              <td v-for="(cell, j) in row" :key="j" :class="{ 'risk-cell-td': predictResult.columns[j] === '风险评分' }">
                {{ typeof cell === 'number' ? fmt(cell, Number(cell) < 1 ? 4 : 2) : cell }}
              </td>
            </tr>
          </tbody>
        </table>
        <p v-if="predictResult.rows.length > 100" style="color:var(--text-dim);padding:8px;font-size:12px">
          显示前 100 条, 共 {{ predictResult.total }} 条。下载 CSV 获取完整结果。
        </p>
      </div>
    </section>

    <!-- 逐站点精度对比表 -->
    <section v-if="bySiteData.length" class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">逐站点汇总</p>
          <h3>五个风场预测精度一览</h3>
          <p>每个站点在验证集上的预测精度指标全面对比。</p>
        </div>
      </div>
      <div class="pred-site-table">
        <div class="pred-table-row head">
          <span>站点</span><span>MAE</span><span>RMSE</span><span>R²</span><span>实际均值</span><span>预测均值</span>
        </div>
        <div v-for="s in bySiteData" :key="s.site_id" class="pred-table-row">
          <strong>{{ s.site_id.toUpperCase() }}</strong>
          <span>{{ fmt(s.mae_mw) }}</span>
          <span>{{ fmt(s.rmse_mw) }}</span>
          <span>{{ fmt(s.r2, 4) }}</span>
          <span>{{ fmt(s.actual_mean_mw) }}</span>
          <span>{{ fmt(s.pred_mean_mw) }}</span>
        </div>
      </div>
    </section>
  </div>
</template>
