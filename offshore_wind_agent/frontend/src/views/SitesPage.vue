<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useDashboard } from '../composables/useDashboard'
import FujianMapCard from '../components/FujianMapCard.vue'
import PowerChart from '../components/PowerChart.vue'

const route = useRoute()

const {
  stationOverview,
  currentSite,
  currentSiteId,
  dailySummary,
  siteTopWindows,
  uploadList,
  ensureDashboardLoaded,
  ensureSiteLoaded,
  loadUploadList,
  loadUploadSiteData,
  fmt,
  riskTone,
} = useDashboard()

// ---- 数据源切换 ----
const dataSource = ref('training')      // 'training' | 'upload'
const selectedUploadId = ref(null)
const uploadSiteData = ref(null)        // 当前站点的上传数据
const uploadLoading = ref(false)

// 上传模式下的日汇总
const uploadDailySummary = computed(() => uploadSiteData.value?.daily_summary || [])
const uploadAvailableDates = computed(() => uploadDailySummary.value.map(r => r.date))

// 上传模式下的选中日
const uploadDailyRow = computed(() => {
  const target = selectedDate.value || uploadAvailableDates.value[0]
  return uploadDailySummary.value.find(r => r.date === target) || null
})

const uploadDailyActions = computed(() => {
  if (!selectedDate.value || dataSource.value !== 'upload') return []
  const dailyActs = uploadSiteData.value?.daily_actions || []
  const entry = dailyActs.find(d => d.date === selectedDate.value)
  if (!entry) return []
  const total = Object.values(entry.actions).reduce((s, v) => s + v, 0) || 1
  return Object.entries(entry.actions)
    .map(([name, count]) => ({ name, count, pct: Math.round(count / total * 100) }))
    .sort((a, b) => b.count - a.count)
})

const uploadDailyTimeSlots = computed(() => {
  if (!selectedDate.value || dataSource.value !== 'upload') return []
  const datePrefix = selectedDate.value.slice(5)
  return (uploadSiteData.value?.forecast_series || []).filter(
    s => s.timestamp && s.timestamp.startsWith(datePrefix)
  )
})

// 统一入口
const effectiveSiteData = computed(() =>
  dataSource.value === 'upload' ? uploadSiteData.value : currentSite.value
)
const effectiveDailySummary = computed(() =>
  dataSource.value === 'upload' ? uploadDailySummary.value : dailySummary.value
)
const effectiveDailyRow = computed(() =>
  dataSource.value === 'upload' ? uploadDailyRow.value : selectedDailyRow.value
)
const effectiveDailyActions = computed(() =>
  dataSource.value === 'upload' ? uploadDailyActions.value : dailyActions.value
)
const effectiveDailyTimeSlots = computed(() =>
  dataSource.value === 'upload' ? uploadDailyTimeSlots.value : dailyTimeSlots.value
)
const effectiveAvailableDates = computed(() =>
  dataSource.value === 'upload' ? uploadAvailableDates.value : availableDates.value
)

async function switchToUpload(uploadId, siteId) {
  dataSource.value = 'upload'
  selectedUploadId.value = uploadId
  uploadLoading.value = true
  try {
    if (siteId) currentSiteId.value = siteId
    const data = await loadUploadSiteData(uploadId, currentSiteId.value)
    uploadSiteData.value = data
    selectedDate.value = uploadAvailableDates.value[0] || ''
  } finally {
    uploadLoading.value = false
  }
}

async function switchToTraining() {
  dataSource.value = 'training'
  selectedUploadId.value = null
  uploadSiteData.value = null
  await ensureSiteLoaded(currentSiteId.value)
  selectedDate.value = availableDates.value[0] || ''
}

const searchKeyword = ref('')
const riskFilter = ref('all')
const chartMode = ref('compare')
const drawerOpen = ref(false)
const hoveredWindow = ref(null)
const selectedDate = ref('')
const datePage = ref(0)
const DATES_PER_PAGE = 14

const filteredStations = computed(() => {
  return stationOverview.value.filter((site) => {
    const keyword = searchKeyword.value.trim().toLowerCase()
    const keywordMatch =
      !keyword ||
      site.site_name.toLowerCase().includes(keyword) ||
      site.site_id.toLowerCase().includes(keyword)
    const tone = riskTone(site.avg_risk)
    const riskMatch = riskFilter.value === 'all' || tone === riskFilter.value
    return keywordMatch && riskMatch
  })
})

const availableDates = computed(() => dailySummary.value.map((row) => row.date))

const selectedDailyRow = computed(() => {
  const target = selectedDate.value || availableDates.value[0]
  return dailySummary.value.find((row) => row.date === target) || null
})

const riskLevelLabel = computed(() => {
  if (!effectiveDailyRow.value) return '--'
  const tone = riskTone(effectiveDailyRow.value.avg_risk)
  if (tone === 'high') return '高风险日'
  if (tone === 'medium') return '中风险日'
  return '低风险日'
})

const dailyComparison = computed(() => {
  if (!selectedDailyRow.value || dataSource.value === 'upload') return null
  const row = selectedDailyRow.value
  const actualEnergy = row.total_actual_mwh || (row.actual_avg * 24 * 0.25)
  const predictedEnergy = row.total_predicted_mwh || (row.predicted_avg * 24 * 0.25)
  return {
    actual: fmt(actualEnergy),
    predicted: fmt(predictedEnergy),
    diff: fmt(actualEnergy - predictedEnergy),
  }
})

const dailyActions = computed(() => {
  if (!selectedDate.value) return []
  const dateSeries = (currentSite.value?.forecast_series || []).filter(
    s => s.timestamp && s.timestamp.startsWith(selectedDate.value.slice(5))
  )
  const counts = {}
  dateSeries.forEach(s => {
    const label = s.action_label || '未知'
    counts[label] = (counts[label] || 0) + 1
  })
  const total = dateSeries.length || 1
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count, pct: Math.round(count / total * 100) }))
    .sort((a, b) => b.count - a.count)
})

const dailyTimeSlots = computed(() => {
  if (!selectedDate.value) return []
  const datePrefix = selectedDate.value.slice(5)
  return (currentSite.value?.forecast_series || []).filter(
    s => s.timestamp && s.timestamp.startsWith(datePrefix)
  )
})

watch(filteredStations, async (sites) => {
  if (!sites.length) return
  if (!sites.some((site) => site.site_id === currentSiteId.value)) {
    if (dataSource.value === 'upload' && selectedUploadId.value) {
      await switchToUpload(selectedUploadId.value, sites[0].site_id)
    } else {
      await ensureSiteLoaded(sites[0].site_id)
    }
  }
})

watch(currentSiteId, async (newId) => {
  if (dataSource.value === 'upload' && selectedUploadId.value && newId) {
    uploadLoading.value = true
    try {
      const data = await loadUploadSiteData(selectedUploadId.value, newId)
      uploadSiteData.value = data
      selectedDate.value = uploadAvailableDates.value[0] || ''
    } finally {
      uploadLoading.value = false
    }
  }
})

watch(availableDates, (dates) => {
  if (!dates.length) { selectedDate.value = ''; return }
  if (!dates.includes(selectedDate.value)) selectedDate.value = dates[0]
})

watch(uploadAvailableDates, (dates) => {
  if (dataSource.value !== 'upload' || !dates.length) return
  if (!dates.includes(selectedDate.value)) selectedDate.value = dates[0]
})

onMounted(async () => {
  await ensureDashboardLoaded()
  await loadUploadList()

  // 检查 URL query 参数
  if (route.query.source === 'upload' && route.query.uploadId) {
    const uploadId = route.query.uploadId
    const siteId = route.query.site || stationOverview.value[0]?.site_id || ''
    await switchToUpload(uploadId, siteId)
    return
  }

  if (currentSiteId.value) {
    await ensureSiteLoaded(currentSiteId.value)
  }
  selectedDate.value = availableDates.value[0] || ''
})

// 分页日期列表
const paginatedDates = computed(() => {
  const all = effectiveAvailableDates.value
  const start = datePage.value * DATES_PER_PAGE
  return all.slice(start, start + DATES_PER_PAGE)
})
const totalDatePages = computed(() => Math.ceil(effectiveAvailableDates.value.length / DATES_PER_PAGE))
const canPrevDatePage = computed(() => datePage.value > 0)
const canNextDatePage = computed(() => datePage.value < totalDatePages.value - 1)

function openDrawer() {
  drawerOpen.value = true
}

function closeDrawer() {
  drawerOpen.value = false
}
</script>

<template>
  <div class="page-grid">
    <!-- 数据源切换 -->
    <section class="page-panel filter-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">站点驾驶舱</p>
          <h3>
            数据源:
            <span v-if="dataSource === 'training'" style="color:var(--aqua)">训练验证数据</span>
            <span v-else style="color:var(--amber)">上传测试数据</span>
          </h3>
          <p v-if="dataSource === 'training'">当前使用训练时的验证集数据，含实际功率可进行预测 vs 实际对比。</p>
          <p v-else>当前使用上传的测试集预测结果（{{ effectiveSiteData?.data_source || '' }}），仅有预测值无实际功率对比。</p>
        </div>
        <div class="chip-group">
          <button type="button" class="mode-chip" :class="{ active: dataSource === 'training' }"
            :disabled="uploadLoading" @click="switchToTraining()">
            训练验证数据
          </button>
          <select v-if="uploadList.length" v-model="selectedUploadId"
            @change="switchToUpload(selectedUploadId, currentSiteId)"
            class="mode-chip" style="max-width:260px">
            <option value="" disabled>选择上传文件...</option>
            <option v-for="u in uploadList" :key="u.id" :value="u.id"
              :selected="u.id === selectedUploadId">
              {{ u.original_filename?.slice(0, 30) }} ({{ u.created_at?.slice(0, 10) }})
            </option>
          </select>
          <span v-else class="mode-chip" style="opacity:0.5">暂无上传数据</span>
        </div>
      </div>
      <div class="filter-row">
        <input v-model="searchKeyword" class="filter-input" type="text" placeholder="搜索站点名称或站点编号" />
        <select v-model="riskFilter" :disabled="dataSource === 'upload'">
          <option value="all">全部风险等级</option>
          <option value="high">高风险</option>
          <option value="medium">中风险</option>
          <option value="low">低风险</option>
        </select>
      </div>
      <div v-if="uploadLoading" style="padding:8px;color:var(--text-dim)">加载上传数据中...</div>
      <div class="station-list-grid">
        <button
          v-for="site in filteredStations"
          :key="site.site_id"
          type="button"
          class="station-summary-card"
          :class="{ active: site.site_id === currentSiteId }"
          @click="ensureSiteLoaded(site.site_id)"
        >
          <span>{{ site.region }}</span>
          <strong>{{ site.site_name }}</strong>
          <small>{{ site.site_id.toUpperCase() }} · {{ fmt(site.capacity_mw, 0) }} MW · MAE {{ fmt(site.site_mae_mw) }} MW</small>
          <div class="fleet-tags">
            <span>风险 {{ fmt(site.avg_risk, 3) }}</span>
            <span>{{ site.dominant_action }}</span>
          </div>
        </button>
      </div>
    </section>

    <!-- 地图 + 站点验证曲线 -->
    <section class="dual-grid">
      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">福建风场地图</p>
            <h3>福建风电场分布</h3>
            <p>用发光联动地图展示站点位置与风险状态。</p>
          </div>
        </div>
        <FujianMapCard
          :stations="filteredStations"
          :current-site-id="currentSiteId"
          :risk-tone="riskTone"
          :fmt="fmt"
          :selected-site="currentSite"
        />
      </article>

      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">{{ dataSource === 'training' ? '验证集曲线' : '预测曲线' }}</p>
            <h3>
              {{ effectiveSiteData?.site_name || currentSite?.site_name || '选择站点' }} —
              {{ dataSource === 'training' ? '预测 vs 实际功率' : '预测功率' }}
            </h3>
            <p v-if="dataSource === 'training'">绿色=实际功率，青色=模型输出，橙色=调度建议。</p>
            <p v-else>青色=模型预测功率，橙色=调度建议。无可比对的真实功率数据。</p>
          </div>
          <div class="chip-group">
            <button type="button" class="mode-chip" :class="{ active: chartMode === 'compare' }"
              @click="chartMode = 'compare'">{{ dataSource === 'training' ? '预测vs实际' : '预测功率' }}</button>
            <button type="button" class="mode-chip" :class="{ active: chartMode === 'dispatch' }" @click="chartMode = 'dispatch'">调度聚焦</button>
            <button type="button" class="mode-chip" :class="{ active: chartMode === 'risk' }" @click="chartMode = 'risk'">风险走势</button>
          </div>
        </div>
        <PowerChart
          :series="effectiveSiteData?.forecast_series || currentSite?.forecast_series || []"
          :fmt="fmt"
          :view-mode="chartMode"
          :validation-mode="dataSource === 'training'"
        />
      </article>
    </section>

    <!-- 日汇总 + 高风险窗口 -->
    <section class="dual-grid">
      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">日期检索</p>
            <h3>{{ dataSource === 'training' ? '验证集日汇总 — 预测 vs 实际' : '预测日汇总' }}</h3>
            <p>共 {{ effectiveAvailableDates.length }} 天数据。选择一个日期查看{{ dataSource === 'training' ? '实际与预测对比' : '预测详情' }}。</p>
          </div>
        </div>

        <div v-if="effectiveDailyRow" class="daily-focus-layout">
          <article class="daily-focus-card">
            <div class="daily-card-top">
              <strong>{{ effectiveDailyRow.date }}</strong>
              <span class="risk-pill" :class="riskTone(effectiveDailyRow.avg_risk)">{{ riskLevelLabel }}</span>
            </div>
            <div class="daily-focus-metrics">
              <div v-if="dataSource === 'training'">
                <span>实际均值</span>
                <strong>{{ fmt(effectiveDailyRow.actual_avg) }} MW</strong>
              </div>
              <div>
                <span>预测均值</span>
                <strong>{{ fmt(effectiveDailyRow.predicted_avg) }} MW</strong>
              </div>
              <div v-if="dataSource === 'training'">
                <span>实际峰值</span>
                <strong>{{ fmt(effectiveDailyRow.actual_peak) }} MW</strong>
              </div>
              <div>
                <span>预测峰值</span>
                <strong>{{ fmt(effectiveDailyRow.predicted_peak) }} MW</strong>
              </div>
              <div v-if="dataSource === 'training'">
                <span>当日实际电量</span>
                <strong>{{ dailyComparison?.actual }} MWh</strong>
              </div>
              <div>
                <span>当日{{ dataSource === 'training' ? '预测' : '预估' }}电量</span>
                <strong>{{ fmt(effectiveDailyRow.total_predicted_mwh || effectiveDailyRow.predicted_avg * 24 * 0.25) }} MWh</strong>
              </div>
              <div v-if="dataSource === 'training'">
                <span>平均误差</span>
                <strong>{{ fmt(effectiveDailyRow.avg_error) }} MW</strong>
              </div>
              <div v-if="dataSource === 'training'">
                <span>电量偏差</span>
                <strong>{{ dailyComparison?.diff }} MWh</strong>
              </div>
              <div>
                <span>平均风险</span>
                <strong>{{ fmt(effectiveDailyRow.avg_risk, 3) }}</strong>
              </div>
            </div>
            <!-- 当日策略分布 -->
            <div v-if="effectiveDailyActions.length" class="daily-action-strip">
              <span class="eyebrow" style="margin-bottom:6px">当日调度策略分布</span>
              <div class="action-bar-row">
                <span v-for="a in effectiveDailyActions" :key="a.name" class="action-pct-chip">{{ a.name }} {{ a.pct }}%</span>
              </div>
            </div>
          </article>

          <!-- 分页日期导航 -->
          <div class="date-nav-bar">
            <button type="button" class="mode-chip" :disabled="!canPrevDatePage" @click="datePage--">◀ 前14天</button>
            <span class="range-label">第 {{ datePage + 1 }}/{{ totalDatePages }} 页</span>
            <button type="button" class="mode-chip" :disabled="!canNextDatePage" @click="datePage++">后14天 ▶</button>
          </div>
          <div class="daily-mini-calendar">
            <button
              v-for="date in paginatedDates"
              :key="date"
              type="button"
              class="mini-date-chip"
              :class="{ active: selectedDate === date }"
              @click="selectedDate = date"
            >
              {{ date.slice(5) }}
            </button>
          </div>
        </div>
      </article>

      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">时段调度</p>
            <h3>{{ selectedDate || '--' }} — 全时段调度建议</h3>
            <p>当天所有时间点的调度策略。红色背景 = 高风险时段（风险≥0.45），建议保守操作。</p>
          </div>
        </div>
        <div v-if="!effectiveDailyTimeSlots.length" style="padding:16px;color:var(--text-dim)">
          暂无数据。
        </div>
        <div v-else class="timeslot-list">
          <div
            v-for="row in effectiveDailyTimeSlots"
            :key="row.timestamp"
            class="timeslot-row"
            :class="{ 'high-risk': Number(row.risk_score) >= 0.45 }"
            @mouseenter="hoveredWindow = row"
            @mouseleave="hoveredWindow = null"
          >
            <strong>{{ (row.timestamp || '').slice(6) }}</strong>
            <span class="timeslot-power" v-if="dataSource === 'training'">实{{ fmt(row.actual_power_mw) }} / 预{{ fmt(row.predicted_power_mw) }} MW</span>
            <span class="timeslot-power" v-else>预{{ fmt(row.predicted_power_mw) }} MW</span>
            <span class="risk-pill" :class="riskTone(row.risk_score)">{{ fmt(row.risk_score, 2) }}</span>
            <span class="timeslot-action">{{ row.action_label }}</span>
          </div>
        </div>
        <div v-if="hoveredWindow" class="hover-detail-card">
          <span>详情</span>
          <strong>{{ hoveredWindow.timestamp }}</strong>
          <small v-if="dataSource === 'training'">
            风险 {{ fmt(hoveredWindow.risk_score, 3) }} ·
            实际 {{ fmt(hoveredWindow.actual_power_mw) }} MW ·
            预测 {{ fmt(hoveredWindow.predicted_power_mw) }} MW ·
            误差 {{ fmt(Math.abs(hoveredWindow.actual_power_mw - hoveredWindow.predicted_power_mw), 3) }} MW ·
            {{ hoveredWindow.action_label }}
          </small>
          <small v-else>
            风险 {{ fmt(hoveredWindow.risk_score, 3) }} ·
            预测 {{ fmt(hoveredWindow.predicted_power_mw) }} MW ·
            {{ hoveredWindow.action_label }}
          </small>
        </div>
      </article>
    </section>

    <!-- 站点详情抽屉 -->
    <transition name="drawer-fade">
      <aside v-if="drawerOpen && effectiveSiteData" class="site-drawer-mask" @click.self="closeDrawer">
        <div class="site-drawer">
          <div class="panel-head">
            <div>
              <p class="eyebrow">站点详情抽屉</p>
              <h3>{{ currentSite?.site_name || effectiveSiteData.site_id?.toUpperCase() }}</h3>
              <p>{{ effectiveSiteData.data_source || '验证集数据，含真实功率可对比' }}</p>
            </div>
            <button type="button" class="button ghost" @click="closeDrawer">关闭</button>
          </div>

          <div class="drawer-metrics">
            <article class="drawer-tile">
              <span>装机容量</span>
              <strong>{{ fmt(effectiveSiteData.capacity_mw, 0) }} MW</strong>
            </article>
            <article v-if="dataSource === 'training'" class="drawer-tile">
              <span>站点 MAE</span>
              <strong>{{ fmt(effectiveSiteData.site_mae_mw) }} MW</strong>
            </article>
            <article v-if="dataSource === 'training'" class="drawer-tile">
              <span>实际均值</span>
              <strong>{{ fmt(effectiveSiteData.actual_mean_mw) }} MW</strong>
            </article>
            <article class="drawer-tile">
              <span>预测均值</span>
              <strong>{{ fmt(effectiveSiteData.predicted_mean_mw || '--') }} MW</strong>
            </article>
            <article class="drawer-tile">
              <span>平均风险</span>
              <strong>{{ fmt(effectiveSiteData.avg_risk, 3) }}</strong>
            </article>
            <article class="drawer-tile">
              <span>主导动作</span>
              <strong>{{ effectiveSiteData.dominant_action }}</strong>
            </article>
          </div>

          <div class="drawer-section">
            <span>当日调度摘要 ({{ selectedDate || '--' }})</span>
            <div class="summary-stack">
              <article v-for="row in effectiveDailyTimeSlots.filter(r => Number(r.risk_score) >= 0.45).slice(0, 4)"
                :key="row.timestamp" class="summary-line-card">
                <span>{{ row.timestamp }}</span>
                <strong>{{ row.action_label }}</strong>
                <small v-if="dataSource === 'training'">风险 {{ fmt(row.risk_score, 3) }} · 实际 {{ fmt(row.actual_power_mw) }} MW · 预测 {{ fmt(row.predicted_power_mw) }} MW</small>
                <small v-else>风险 {{ fmt(row.risk_score, 3) }} · 预测 {{ fmt(row.predicted_power_mw) }} MW</small>
              </article>
            </div>
          </div>
        </div>
      </aside>
    </transition>
  </div>
</template>
