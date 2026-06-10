<script setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useDashboard } from '../composables/useDashboard'
import PowerChart from '../components/PowerChart.vue'

const router = useRouter()

const {
  currentSite, stationOverview, currentSiteId,
  rlComparison, topRiskWindows, summaryCards,
  validationComparison, uploadList,
  ensureDashboardLoaded, ensureSiteLoaded, loadUploadList,
  fmt, riskColor, riskTone,
} = useDashboard()

const rlMaxReward = computed(() =>
  Math.max(1, ...rlComparison.value.map(e => Number(e.avg_reward) || 0)),
)
const validationBySite = computed(() => validationComparison.value?.by_site || [])
const validationOverall = computed(() => ({
  mae: validationComparison.value?.overall_mae_mw || 0,
  rmse: validationComparison.value?.overall_rmse_mw || 0,
}))
const errorDist = computed(() => validationComparison.value?.error_distribution || {})

function goToPredictions() {
  router.push('/prediction')
}
function goToUploadSite(uploadId, siteId) {
  router.push({ path: '/sites', query: { source: 'upload', uploadId, site: siteId } })
}

onMounted(async () => {
  await ensureDashboardLoaded()
  await loadUploadList()
  if (currentSiteId.value) await ensureSiteLoaded(currentSiteId.value)
})
</script>

<template>
  <div class="page-grid">
    <!-- 上传数据指示条 -->
    <section v-if="uploadList.length" class="page-panel wide-panel upload-indicator-bar">
      <div class="upload-indicator-content">
        <span style="font-weight:600;color:var(--amber)">已加载测试集</span>
        <span v-for="u in uploadList.slice(0, 3)" :key="u.id" class="upload-indicator-item"
          @click="goToPredictions()" style="cursor:pointer">
          {{ u.original_filename?.slice(0, 35) }}{{ u.original_filename?.length > 35 ? '...' : '' }}
          <small>({{ u.total_rows }}条, {{ u.sites?.length }}站点)</small>
        </span>
        <button class="mode-chip" @click="goToPredictions()">查看全部</button>
      </div>
    </section>

    <!-- 顶部指标 -->
    <section class="top-strip">
      <article v-for="card in summaryCards" :key="card.label" class="page-panel metric-tile">
        <span>{{ card.label }}</span>
        <strong>{{ card.value }}</strong>
        <small>{{ card.note }}</small>
      </article>
    </section>

    <!-- 验证集精度 + 误差分布 -->
    <section v-if="validationComparison" class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">精度总览</p>
          <h3>验证集模型精度 — MAE {{ fmt(validationOverall.mae) }} MW / RMSE {{ fmt(validationOverall.rmse) }} MW</h3>
          <p>误差 &lt; 1MW 占比 {{ errorDist.within_1mw_pct || '--' }}% · &lt; 3MW 占比 {{ errorDist.within_3mw_pct || '--' }}% · 均值 {{ fmt(errorDist.mean) }} MW · P95 {{ fmt(errorDist.p95) }} MW</p>
        </div>
      </div>
      <div class="error-bar-row">
        <div class="error-seg good" :style="{flex: errorDist.within_1mw_pct || 0}"><small>&lt;1MW<br>{{ errorDist.within_1mw_pct }}%</small></div>
        <div class="error-seg ok" :style="{flex: Math.max(0,(errorDist.within_3mw_pct||0)-(errorDist.within_1mw_pct||0))}"><small>1-3MW<br>{{ fmt(Math.max(0,(errorDist.within_3mw_pct||0)-(errorDist.within_1mw_pct||0)),1) }}%</small></div>
        <div class="error-seg bad" :style="{flex: Math.max(1,100-(errorDist.within_3mw_pct||0))}"><small>&gt;3MW<br>{{ fmt(Math.max(0,100-(errorDist.within_3mw_pct||0)),1) }}%</small></div>
      </div>
      <div class="validation-site-grid" style="margin-top:16px">
        <article v-for="s in validationBySite" :key="s.site_id" class="validation-site-card">
          <div class="val-site-head"><strong>{{ s.site_id.toUpperCase() }}</strong><span class="mae-badge">MAE {{ fmt(s.mae_mw) }} MW</span></div>
          <div class="val-site-metrics">
            <div><span>RMSE</span><strong>{{ fmt(s.rmse_mw) }}</strong></div>
            <div><span>MAPE</span><strong>{{ fmt(s.mape_pct) }}%</strong></div>
            <div><span>实际均值</span><strong>{{ fmt(s.actual_mean_mw) }}</strong></div>
            <div><span>预测均值</span><strong>{{ fmt(s.pred_mean_mw) }}</strong></div>
          </div>
        </article>
      </div>
    </section>

    <!-- 当前站点验证曲线 -->
    <section class="page-panel wide-panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">站点验证</p>
          <h3>{{ currentSite?.site_name || '--' }} — 预测 vs 实际功率曲线</h3>
          <p>绿色=实际功率 · 青色=模型输出 · 橙色=调度建议。可拖动下方时间窗口查看更多数据。</p>
        </div>
        <div class="data-badge"><span>站点 MAE</span><strong>{{ fmt(currentSite?.site_mae_mw) }} MW</strong><small>验证集有真实功率可对比</small></div>
      </div>
      <PowerChart :series="currentSite?.forecast_series || []" :fmt="fmt" :validation-mode="true" />
    </section>

    <!-- 风场卡片 + 策略对比 -->
    <section class="dual-grid">
      <article class="page-panel">
        <div class="panel-head"><div><p class="eyebrow">风场总览</p><h3>各站点验证指标</h3></div></div>
        <div class="fleet-grid">
          <article v-for="s in stationOverview" :key="s.site_id" class="fleet-card">
            <span>{{ s.region }}</span>
            <strong>{{ s.site_name }}</strong>
            <small>{{ fmt(s.capacity_mw,0) }} MW · MAE {{ fmt(s.site_mae_mw) }} MW</small>
            <small>实际 {{ fmt(s.actual_energy_mwh) }} / 预测 {{ fmt(s.predicted_energy_mwh) }} MWh</small>
            <div class="fleet-tags"><span>风险 {{ fmt(s.avg_risk,3) }}</span><span>{{ s.dominant_action }}</span></div>
          </article>
        </div>
      </article>

      <article class="page-panel">
        <div class="panel-head"><div><p class="eyebrow">策略收益</p><h3>调度策略对比</h3><p>各策略在统一验证环境下的累计收益。越大越好。</p></div></div>
        <div class="rl-bars">
          <div v-for="item in rlComparison" :key="item.policy" class="bar-row">
            <div><strong>{{ item.policy }}</strong><small>{{ item.dominant_action }}</small></div>
            <div class="bar-track"><div class="bar-fill" :style="{width:(Number(item.avg_reward)/rlMaxReward*100)+'%'}"></div></div>
            <div class="bar-side"><strong>{{ fmt(item.avg_reward,2) }}</strong><small>事故率 {{ fmt(item.incident_rate,3) }}</small></div>
          </div>
        </div>
      </article>
    </section>

    <!-- 风险说明 (简化清晰版) -->
    <section class="dual-grid">
      <article class="page-panel flex-col">
        <div class="panel-head">
          <div>
            <p class="eyebrow">风险监控</p>
            <h3>风险热力图</h3>
            <p>每个色块代表一个时间片的风险等级。绿=低风险(安全)，黄=中风险(注意)，红=高风险(需保守操作)。风险越高调度越应保守。</p>
          </div>
        </div>
        <div class="risk-legend-row">
          <span class="risk-dot" style="background:#4caf50"></span>低风险
          <span class="risk-dot" style="background:#ff9800"></span>中风险
          <span class="risk-dot" style="background:#f44336"></span>高风险
        </div>
        <div class="risk-band">
          <div v-for="cell in (currentSite?.risk_band || [])" :key="cell.timestamp" class="risk-cell" :style="{background:riskColor(cell.risk_score)}" :title="`${cell.timestamp} | 风险${fmt(cell.risk_score,3)} | ${cell.action_label}`"></div>
        </div>
      </article>

      <article class="page-panel">
        <div class="panel-head">
          <div>
            <p class="eyebrow">高风险预警</p>
            <h3>风险最高时段 Top 8</h3>
            <p>列出验证集中风险评分最高的时段。实=实际功率，预=模型输出。风险越高越应保守调度。</p>
          </div>
        </div>
        <div class="table-list compact-list">
          <div v-for="row in topRiskWindows.slice(0,8)" :key="`${row.site_id}-${row.timestamp}`" class="table-row">
            <strong>{{ row.site_id }}</strong>
            <span>{{ row.timestamp }}</span>
            <span class="risk-pill" :class="riskTone(row.risk_score)">{{ fmt(row.risk_score,3) }}</span>
            <span>实{{ row.actual_power_mw||'--' }}/预{{ row.predicted_power_mw||'--' }}MW</span>
          </div>
        </div>
        <div class="risk-legend-row" style="margin-top:12px">
          <small>风险评分 = 综合风速突变(28%) + 降水(18%) + 高风速(14%) + 密度异常(12%) + 云量(8%) + 站点历史误差(12%) + 高出力压力(8%)</small>
        </div>
      </article>
    </section>
  </div>
</template>
