import { computed, ref } from 'vue'
import { askAgent, fetchDashboard, fetchSite, getExportUrl, saveUpload, listUploads, getUploadDetail, getUploadSiteData, deleteUpload } from '../services/agentApi'

const dashboard = ref(null)
const siteMap = ref({})
const currentSiteId = ref('')
const loading = ref(false)
const siteLoading = ref(false)
const asking = ref(false)
const errorMessage = ref('')
const answer = ref('等待提问...')
const answerMeta = ref('')

// ---- 上传测试集状态 ----
const uploadList = ref([])
const currentUploadId = ref(null)
const currentUpload = ref(null)
const uploadSiteDataCache = ref({})   // key: "<uploadId>/<siteId>"
const uploading = ref(false)
const uploadResult = ref(null)        // 刚上传完成的结果 (含 columns + rows)

let dashboardPromise = null

const stationOverview = computed(() => dashboard.value?.station_overview || [])
const currentSite = computed(() => siteMap.value[currentSiteId.value] || null)
const summaryCards = computed(() => dashboard.value?.summary_cards || [])
const rlComparison = computed(() => dashboard.value?.rl_comparison || [])
const experimentModules = computed(() => dashboard.value?.experiment_modules || [])
const heuristicExamples = computed(() => dashboard.value?.heuristic_examples || [])
const gaExamples = computed(() => dashboard.value?.ga_examples || [])
const topRiskWindows = computed(() => dashboard.value?.top_risk_windows || [])
const actionCatalog = computed(() => dashboard.value?.action_catalog || [])
const agentBackend = computed(() => dashboard.value?.agent_backend || null)
const modelComparison = computed(() => dashboard.value?.model_comparison || {})
const rlAlgorithmComparison = computed(() => dashboard.value?.rl_algorithm_comparison || {})
const dailySummary = computed(() => currentSite.value?.daily_summary || [])
const siteTopWindows = computed(() => currentSite.value?.top_windows || [])
const exportForecastUrl = computed(() => getExportUrl('/export/forecast.csv'))
const exportReportUrl = computed(() => getExportUrl('/export/rl_report.json'))
const activeStation = computed(
  () => stationOverview.value.find((item) => item.site_id === currentSiteId.value) || null,
)

// ---- 新增: 验证集预测 vs 实际对比数据 ----
const validationComparison = computed(() => dashboard.value?.validation_comparison || null)
const algorithmSimulation = computed(() => dashboard.value?.algorithm_simulation || null)
const dataTimeline = computed(() => dashboard.value?.data_timeline || null)

function fmt(value, digits = 2) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : '--'
}

function riskTone(score) {
  const numeric = Number(score) || 0
  if (numeric >= 0.75) return 'high'
  if (numeric >= 0.45) return 'medium'
  return 'low'
}

function riskColor(score) {
  const safeScore = Math.max(0, Math.min(Number(score) || 0, 1))
  const hue = 148 - Math.round(safeScore * 138)
  return `hsl(${hue} 72% 48%)`
}

async function ensureDashboardLoaded(force = false) {
  if (dashboard.value && !force) return dashboard.value
  if (dashboardPromise && !force) return dashboardPromise

  loading.value = true
  errorMessage.value = ''
  dashboardPromise = fetchDashboard()
    .then(async (payload) => {
      dashboard.value = payload
      const firstSiteId = payload.station_overview?.[0]?.site_id || ''
      if (!currentSiteId.value && firstSiteId) {
        currentSiteId.value = firstSiteId
      }
      if (currentSiteId.value) {
        await ensureSiteLoaded(currentSiteId.value)
      }
      return payload
    })
    .catch((error) => {
      errorMessage.value = `加载仪表盘失败：${error instanceof Error ? error.message : '未知错误'}`
      throw error
    })
    .finally(() => {
      loading.value = false
      dashboardPromise = null
    })

  return dashboardPromise
}

async function ensureSiteLoaded(siteId) {
  if (!siteId) return null
  currentSiteId.value = siteId
  if (siteMap.value[siteId]) return siteMap.value[siteId]

  siteLoading.value = true
  try {
    const payload = await fetchSite(siteId)
    siteMap.value = {
      ...siteMap.value,
      [siteId]: payload,
    }
    return payload
  } catch (error) {
    errorMessage.value = `站点详情加载失败：${error instanceof Error ? error.message : '未知错误'}`
    throw error
  } finally {
    siteLoading.value = false
  }
}

async function submitQuestion(question) {
  const text = String(question || '').trim()
  if (!text) {
    answer.value = '请先输入问题。'
    answerMeta.value = ''
    return null
  }

  asking.value = true
  answer.value = '智能体正在分析问题，请稍候...'
  answerMeta.value = ''
  try {
    const payload = await askAgent(text)
    answer.value = payload.answer || '暂无回答。'
    answerMeta.value =
      payload.backend === 'ollama'
        ? `本次回答来自本地模型 ${payload.model || ''}`.trim()
        : '本次回答来自内置规则问答'
    return payload
  } catch (error) {
    answer.value = `提问失败：${error instanceof Error ? error.message : '未知错误'}`
    answerMeta.value = ''
    throw error
  } finally {
    asking.value = false
  }
}

// ---- 上传测试集操作方法 ----

async function loadUploadList() {
  try {
    uploadList.value = await listUploads()
  } catch (e) {
    console.error('loadUploadList failed:', e)
  }
}

async function loadUploadDetail(uploadId) {
  currentUploadId.value = uploadId
  try {
    currentUpload.value = await getUploadDetail(uploadId)
  } catch (e) {
    console.error('loadUploadDetail failed:', e)
    currentUpload.value = null
  }
}

async function loadUploadSiteData(uploadId, siteId) {
  const key = `${uploadId}/${siteId}`
  if (uploadSiteDataCache.value[key]) return uploadSiteDataCache.value[key]
  try {
    const data = await getUploadSiteData(uploadId, siteId)
    uploadSiteDataCache.value = { ...uploadSiteDataCache.value, [key]: data }
    return data
  } catch (e) {
    console.error('loadUploadSiteData failed:', e)
    return null
  }
}

async function saveCurrentUpload(csvContent, originalFilename) {
  uploading.value = true
  try {
    const meta = await saveUpload(csvContent, originalFilename)
    await loadUploadList()
    return meta
  } catch (e) {
    console.error('saveUpload failed:', e)
    throw e
  } finally {
    uploading.value = false
  }
}

async function removeUpload(uploadId) {
  try {
    await deleteUpload(uploadId)
    await loadUploadList()
    if (currentUploadId.value === uploadId) {
      currentUploadId.value = null
      currentUpload.value = null
    }
  } catch (e) {
    console.error('removeUpload failed:', e)
  }
}

export function useDashboard() {
  return {
    dashboard,
    siteMap,
    currentSiteId,
    loading,
    siteLoading,
    asking,
    errorMessage,
    answer,
    answerMeta,
    stationOverview,
    currentSite,
    summaryCards,
    rlComparison,
    experimentModules,
    heuristicExamples,
    gaExamples,
    topRiskWindows,
    actionCatalog,
    agentBackend,
    modelComparison,
    rlAlgorithmComparison,
    dailySummary,
    siteTopWindows,
    exportForecastUrl,
    exportReportUrl,
    activeStation,
    validationComparison,
    algorithmSimulation,
    dataTimeline,
    fmt,
    riskTone,
    riskColor,
    ensureDashboardLoaded,
    ensureSiteLoaded,
    submitQuestion,
    // upload
    uploadList,
    currentUploadId,
    currentUpload,
    uploadSiteDataCache,
    uploading,
    uploadResult,
    loadUploadList,
    loadUploadDetail,
    loadUploadSiteData,
    saveCurrentUpload,
    removeUpload,
  }
}
