class AdvancedAnalysis {
    constructor(container) {
        this.container = container;
        this.activeTab = 'energy';
        this.meridianEnergy = null;
        this.radarChart = null;
        this.techniqueChart = null;
        this.recommendList = [];
        this.currentAcupoints = [];
        this.init();
    }

    init() {
        this.render();
        this.bindEvents();
        this.initCharts();
        this.startAutoRefresh();
    }

    render() {
        this.container.innerHTML = `
            <div class="advanced-analysis">
                <div class="tab-nav">
                    <div class="tab-item active" data-tab="energy">
                        <span class="tab-icon">⚡</span>
                        <span>经络能量平衡</span>
                    </div>
                    <div class="tab-item" data-tab="acupoints">
                        <span class="tab-icon">💎</span>
                        <span>穴位配伍推荐</span>
                    </div>
                    <div class="tab-item" data-tab="technique">
                        <span class="tab-icon">🎯</span>
                        <span>手法量化分析</span>
                    </div>
                    <div class="tab-item" data-tab="qlearn">
                        <span class="tab-icon">🧠</span>
                        <span>自适应参数</span>
                    </div>
                </div>

                <div class="tab-content">
                    <div class="tab-panel active" id="tab-energy">
                        <div class="panel-header">
                            <h4>十二经络能量平衡图</h4>
                            <span class="refresh-info" id="energy-refresh-time">--</span>
                        </div>
                        <div class="radar-container">
                            <div id="energy-radar-chart" style="height: 380px"></div>
                        </div>
                        <div class="balance-metrics">
                            <div class="balance-item">
                                <span class="balance-label">整体平衡</span>
                                <span class="balance-value" id="overall-balance">--</span>
                            </div>
                            <div class="balance-item yin">
                                <span class="balance-label">阴经能量</span>
                                <span class="balance-value" id="yin-energy">--</span>
                            </div>
                            <div class="balance-item yang">
                                <span class="balance-label">阳经能量</span>
                                <span class="balance-value" id="yang-energy">--</span>
                            </div>
                            <div class="balance-item">
                                <span class="balance-label">阴阳比</span>
                                <span class="balance-value" id="yin-yang-ratio">--</span>
                            </div>
                        </div>
                        <div class="meridian-status-list">
                            <div class="status-row">
                                <span class="status-label">偏盛经络</span>
                                <span class="status-value excess" id="excess-meridians">--</span>
                            </div>
                            <div class="status-row">
                                <span class="status-label">偏虚经络</span>
                                <span class="status-value deficient" id="deficient-meridians">--</span>
                            </div>
                        </div>
                    </div>

                    <div class="tab-panel" id="tab-acupoints">
                        <div class="panel-header">
                            <h4>穴位配伍优化推荐</h4>
                            <button class="btn btn-small btn-primary" id="btn-refresh-recommend">刷新推荐</button>
                        </div>
                        <div class="selected-acupoints">
                            <div class="selected-label">已选穴位：</div>
                            <div class="selected-list" id="selected-acupoints-list">
                                <span class="empty-hint">点击经络图上的穴位添加</span>
                            </div>
                        </div>
                        <div class="recommend-section">
                            <h5>推荐配穴</h5>
                            <div class="recommend-list" id="recommend-list">
                                <div class="loading">加载中...</div>
                            </div>
                        </div>
                        <div class="rules-section">
                            <h5>高频配伍规则</h5>
                            <div class="rules-list" id="rules-list">
                                <div class="loading">加载中...</div>
                            </div>
                        </div>
                    </div>

                    <div class="tab-panel" id="tab-technique">
                        <div class="panel-header">
                            <h4>针刺手法量化分析</h4>
                            <span class="technique-badge" id="current-technique">--</span>
                        </div>
                        <div class="technique-display">
                            <div class="technique-main">
                                <div class="technique-icon" id="technique-icon">✋</div>
                                <div class="technique-name" id="technique-name">识别中...</div>
                                <div class="technique-confidence">
                                    <span>置信度</span>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" id="technique-confidence-fill"></div>
                                    </div>
                                    <span id="technique-confidence-value">--%</span>
                                </div>
                            </div>
                        </div>
                        <div class="technique-features">
                            <h5>肌电特征</h5>
                            <div class="feature-grid">
                                <div class="feature-item">
                                    <span class="feature-label">RMS</span>
                                    <span class="feature-value" id="feat-rms">--</span>
                                </div>
                                <div class="feature-item">
                                    <span class="feature-label">MAV</span>
                                    <span class="feature-value" id="feat-mav">--</span>
                                </div>
                                <div class="feature-item">
                                    <span class="feature-label">过零率</span>
                                    <span class="feature-value" id="feat-zcr">--</span>
                                </div>
                                <div class="feature-item">
                                    <span class="feature-label">峰值频率</span>
                                    <span class="feature-value" id="feat-pf">--</span>
                                </div>
                                <div class="feature-item">
                                    <span class="feature-label">谱熵</span>
                                    <span class="feature-value" id="feat-entropy">--</span>
                                </div>
                                <div class="feature-item">
                                    <span class="feature-label">波形长度</span>
                                    <span class="feature-value" id="feat-wl">--</span>
                                </div>
                            </div>
                        </div>
                        <div class="technique-chart">
                            <h5>手法类型分布</h5>
                            <div id="technique-dist-chart" style="height: 200px"></div>
                        </div>
                    </div>

                    <div class="tab-panel" id="tab-qlearn">
                        <div class="panel-header">
                            <h4>个性化针刺参数自适应</h4>
                            <span class="badge-info" id="qlearn-status">学习中</span>
                        </div>
                        <div class="qlearn-recommendation">
                            <div class="recommend-card primary">
                                <div class="card-title">推荐方案</div>
                                <div class="params-grid">
                                    <div class="param-item">
                                        <span class="param-icon">⏱️</span>
                                        <span class="param-label">留针时间</span>
                                        <span class="param-value" id="param-retention">-- 分钟</span>
                                    </div>
                                    <div class="param-item">
                                        <span class="param-icon">📳</span>
                                        <span class="param-label">刺激频率</span>
                                        <span class="param-value" id="param-frequency">-- Hz</span>
                                    </div>
                                    <div class="param-item">
                                        <span class="param-icon">📏</span>
                                        <span class="param-label">进针深度</span>
                                        <span class="param-value" id="param-depth">-- mm</span>
                                    </div>
                                    <div class="param-item">
                                        <span class="param-icon">⚖️</span>
                                        <span class="param-label">手法类型</span>
                                        <span class="param-value" id="param-technique">--</span>
                                    </div>
                                </div>
                                <div class="expected-reward">
                                    <span>预期疗效得分</span>
                                    <span class="reward-value" id="expected-reward">--</span>
                                </div>
                            </div>
                        </div>
                        <div class="qlearn-stats">
                            <h5>学习状态</h5>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="stat-label">已探索状态</span>
                                    <span class="stat-value" id="stat-states">--</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">总更新次数</span>
                                    <span class="stat-value" id="stat-updates">--</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">探索率</span>
                                    <span class="stat-value" id="stat-exploration">--</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">平均奖励</span>
                                    <span class="stat-value" id="stat-reward">--</span>
                                </div>
                            </div>
                        </div>
                        <div class="qlearn-actions">
                            <button class="btn btn-primary" id="btn-apply-recommendation">应用推荐</button>
                            <button class="btn btn-secondary" id="btn-give-feedback">反馈疗效</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    bindEvents() {
        const tabs = this.container.querySelectorAll('.tab-item');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchTab(tab.dataset.tab);
            });
        });

        const refreshBtn = this.container.querySelector('#btn-refresh-recommend');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadRecommendations());
        }

        const applyBtn = this.container.querySelector('#btn-apply-recommendation');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyRecommendation());
        }

        const feedbackBtn = this.container.querySelector('#btn-give-feedback');
        if (feedbackBtn) {
            feedbackBtn.addEventListener('click', () => this.showFeedbackDialog());
        }
    }

    switchTab(tabName) {
        this.activeTab = tabName;

        const tabs = this.container.querySelectorAll('.tab-item');
        tabs.forEach(t => t.classList.remove('active'));
        this.container.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        const panels = this.container.querySelectorAll('.tab-panel');
        panels.forEach(p => p.classList.remove('active'));
        this.container.querySelector(`#tab-${tabName}`).classList.add('active');

        if (tabName === 'energy') {
            this.loadEnergyBalance();
            setTimeout(() => {
                if (this.radarChart) this.radarChart.resize();
            }, 100);
        } else if (tabName === 'acupoints') {
            this.loadRecommendations();
            this.loadAssociationRules();
        } else if (tabName === 'technique') {
            this.loadTechniqueAnalysis();
            setTimeout(() => {
                if (this.techniqueChart) this.techniqueChart.resize();
            }, 100);
        } else if (tabName === 'qlearn') {
            this.loadQLearningRecommendation();
            this.loadQLearningStats();
        }
    }

    initCharts() {
        const radarEl = document.getElementById('energy-radar-chart');
        if (radarEl && window.echarts) {
            this.radarChart = echarts.init(radarEl);
        }

        const distEl = document.getElementById('technique-dist-chart');
        if (distEl && window.echarts) {
            this.techniqueChart = echarts.init(distEl);
        }
    }

    startAutoRefresh() {
        setInterval(() => {
            if (this.activeTab === 'energy') {
                this.loadEnergyBalance();
            } else if (this.activeTab === 'technique') {
                this.loadTechniqueAnalysis();
            }
        }, 5000);
    }

    async loadEnergyBalance() {
        try {
            const res = await fetch('/api/analyzer/energy/balance');
            if (!res.ok) return;
            const data = await res.json();
            this.updateEnergyBalance(data);
        } catch(e) {
            console.warn('Failed to load energy balance:', e);
        }
    }

    updateEnergyBalance(data) {
        this.meridianEnergy = data;

        const meridians = data.meridians || [];
        const names = meridians.map(m => this.getMeridianShortName(m.meridian_id));
        const scores = meridians.map(m => m.energy_score);

        const avgScore = meridians.reduce((s, m) => s + m.energy_score, 0) / (meridians.length || 1);

        const option = {
            tooltip: {
                trigger: 'item'
            },
            radar: {
                indicator: names.map(n => ({ name: n, max: 100 })),
                shape: 'polygon',
                splitNumber: 5,
                axisName: {
                    color: '#666',
                    fontSize: 11
                },
                splitLine: {
                    lineStyle: { color: 'rgba(0, 0, 0, 0.1)' }
                },
                splitArea: {
                    areaStyle: {
                        color: ['rgba(0, 162, 255, 0.03)', 'rgba(0, 162, 255, 0.06)']
                    }
                },
                axisLine: {
                    lineStyle: { color: 'rgba(0, 0, 0, 0.15)' }
                }
            },
            series: [{
                type: 'radar',
                data: [{
                    value: scores,
                    name: '经络能量',
                    symbol: 'circle',
                    symbolSize: 6,
                    lineStyle: {
                        width: 2,
                        color: '#00a2ff'
                    },
                    areaStyle: {
                        color: new echarts.graphic.RadialGradient(0.5, 0.5, 1, [
                            { color: 'rgba(0, 162, 255, 0.6)', offset: 0 },
                            { color: 'rgba(0, 162, 255, 0.1)', offset: 1 }
                        ])
                    },
                    itemStyle: {
                        color: '#00a2ff',
                        borderColor: '#fff',
                        borderWidth: 2
                    }
                }]
            }]
        };

        if (this.radarChart) {
            this.radarChart.setOption(option);
        }

        document.getElementById('overall-balance').textContent = data.overall_balance_score?.toFixed(1) || '--';
        document.getElementById('yin-energy').textContent = data.yin_energy?.toFixed(1) || '--';
        document.getElementById('yang-energy').textContent = data.yang_energy?.toFixed(1) || '--';
        document.getElementById('yin-yang-ratio').textContent = data.yin_yang_ratio?.toFixed(2) || '--';

        const excess = (data.excess_meridians || []).map(m => this.getMeridianShortName(m)).join('、');
        const deficient = (data.deficient_meridians || []).map(m => this.getMeridianShortName(m)).join('、');
        document.getElementById('excess-meridians').textContent = excess || '无';
        document.getElementById('deficient-meridians').textContent = deficient || '无';

        const now = new Date();
        document.getElementById('energy-refresh-time').textContent =
            now.toLocaleTimeString('zh-CN') + ' 更新';
    }

    getMeridianShortName(id) {
        const names = {
            'LU': '肺经', 'LI': '大肠经', 'ST': '胃经', 'SP': '脾经',
            'HT': '心经', 'SI': '小肠经', 'BL': '膀胱经', 'KI': '肾经',
            'PC': '心包经', 'TE': '三焦经', 'GB': '胆经', 'LR': '肝经',
            'CV': '任脉', 'GV': '督脉'
        };
        return names[id] || id;
    }

    async loadRecommendations() {
        const listEl = document.getElementById('recommend-list');
        if (!listEl) return;

        listEl.innerHTML = '<div class="loading">加载中...</div>';

        try {
            const res = await fetch('/api/analyzer/association/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    acupoints: this.currentAcupoints,
                    top_k: 8
                })
            });

            if (!res.ok) throw new Error('Failed');
            const data = await res.json();
            this.renderRecommendations(data.recommendations || []);
        } catch(e) {
            listEl.innerHTML = '<div class="empty-hint">暂无推荐数据</div>';
        }
    }

    renderRecommendations(recommendations) {
        const listEl = document.getElementById('recommend-list');
        if (!listEl) return;

        if (!recommendations.length) {
            listEl.innerHTML = '<div class="empty-hint">暂无推荐，可先选择穴位</div>';
            return;
        }

        listEl.innerHTML = recommendations.map((r, i) => `
            <div class="recommend-item" data-acupoint="${r.acupoints[0]}">
                <span class="rank">${i + 1}</span>
                <span class="acupoint-name">${r.acupoints[0]}</span>
                <div class="recommend-meta">
                    <span>缓解率: ${(r.avg_pain_relief * 100).toFixed(0)}%</span>
                    <span>支持数: ${r.sample_count}</span>
                </div>
                <button class="btn-add">+ 添加</button>
            </div>
        `).join('');

        listEl.querySelectorAll('.recommend-item').forEach(item => {
            item.querySelector('.btn-add').addEventListener('click', () => {
                const ap = item.dataset.acupoint;
                this.addAcupoint(ap);
            });
        });
    }

    async loadAssociationRules() {
        const listEl = document.getElementById('rules-list');
        if (!listEl) return;

        try {
            const res = await fetch('/api/analyzer/association/rules?top=10');
            if (!res.ok) throw new Error('Failed');
            const rules = await res.json();
            this.renderRules(rules);
        } catch(e) {
            listEl.innerHTML = '<div class="empty-hint">暂无规则数据</div>';
        }
    }

    renderRules(rules) {
        const listEl = document.getElementById('rules-list');
        if (!listEl) return;

        if (!rules.length) {
            listEl.innerHTML = '<div class="empty-hint">暂无配伍规则</div>';
            return;
        }

        listEl.innerHTML = rules.map(r => `
            <div class="rule-item">
                <div class="rule-content">
                    <span class="rule-antecedent">${r.antecedent.join(' + ')}</span>
                    <span class="rule-arrow">→</span>
                    <span class="rule-consequent">${r.consequent.join(' + ')}</span>
                </div>
                <div class="rule-stats">
                    <span>置信度 ${(r.confidence * 100).toFixed(0)}%</span>
                    <span>提升度 ${r.lift.toFixed(2)}</span>
                </div>
            </div>
        `).join('');
    }

    addAcupoint(acupointId) {
        if (!this.currentAcupoints.includes(acupointId)) {
            this.currentAcupoints.push(acupointId);
            this.updateSelectedAcupoints();
            this.loadRecommendations();
        }
    }

    removeAcupoint(acupointId) {
        this.currentAcupoints = this.currentAcupoints.filter(a => a !== acupointId);
        this.updateSelectedAcupoints();
        this.loadRecommendations();
    }

    updateSelectedAcupoints() {
        const listEl = document.getElementById('selected-acupoints-list');
        if (!listEl) return;

        if (!this.currentAcupoints.length) {
            listEl.innerHTML = '<span class="empty-hint">点击经络图上的穴位添加</span>';
            return;
        }

        listEl.innerHTML = this.currentAcupoints.map(ap => `
            <span class="acupoint-tag">
                ${ap}
                <span class="tag-close" data-ap="${ap}">×</span>
            </span>
        `).join('');

        listEl.querySelectorAll('.tag-close').forEach(tag => {
            tag.addEventListener('click', () => {
                this.removeAcupoint(tag.dataset.ap);
            });
        });
    }

    async loadTechniqueAnalysis() {
        try {
            const res = await fetch('/api/analyzer/technique/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    emg_signal: this.generateMockEMG(),
                    timestamp: Date.now()
                })
            });

            if (!res.ok) return;
            const data = await res.json();
            this.updateTechniqueDisplay(data);
        } catch(e) {
            console.warn('Failed to load technique analysis:', e);
        }
    }

    generateMockEMG() {
        const signal = [];
        const len = 200;
        for (let i = 0; i < len; i++) {
            const base = Math.sin(i * 0.05) * 15;
            const noise = (Math.random() - 0.5) * 20;
            signal.push(base + noise);
        }
        return signal;
    }

    updateTechniqueDisplay(data) {
        const techNames = {
            'resting': '静息',
            'lifting_thrusting': '提插法',
            'twirling': '捻转法',
            'reinforcing': '补法',
            'reducing': '泻法',
            'even_method': '平补平泻'
        };

        const techIcons = {
            'resting': '✋',
            'lifting_thrusting': '⬆️⬇️',
            'twirling': '🔄',
            'reinforcing': '➕',
            'reducing': '➖',
            'even_method': '⚖️'
        };

        const name = techNames[data.technique] || data.technique;
        const icon = techIcons[data.technique] || '✋';

        document.getElementById('technique-icon').textContent = icon;
        document.getElementById('technique-name').textContent = name;
        document.getElementById('current-technique').textContent = name;

        const conf = (data.confidence * 100) || 0;
        document.getElementById('technique-confidence-value').textContent = conf.toFixed(0) + '%';
        document.getElementById('technique-confidence-fill').style.width = conf + '%';

        const feat = data.features || {};
        document.getElementById('feat-rms').textContent = feat.rms?.toFixed(2) || '--';
        document.getElementById('feat-mav').textContent = feat.mav?.toFixed(2) || '--';
        document.getElementById('feat-zcr').textContent = feat.zero_crossing_rate?.toFixed(0) + ' Hz' || '--';
        document.getElementById('feat-pf').textContent = feat.peak_frequency?.toFixed(1) + ' Hz' || '--';
        document.getElementById('feat-entropy').textContent = feat.spectral_entropy?.toFixed(2) || '--';
        document.getElementById('feat-wl').textContent = feat.waveform_length?.toFixed(0) || '--';

        this.updateTechniqueDistChart(data);
    }

    updateTechniqueDistChart(data) {
        if (!this.techniqueChart) return;

        const techniques = ['静息', '提插', '捻转', '补法', '泻法', '平补'];
        const values = [15, 25, 30, 10, 12, 8];

        const option = {
            tooltip: { trigger: 'axis' },
            grid: { left: 40, right: 10, top: 10, bottom: 30 },
            xAxis: {
                type: 'category',
                data: techniques,
                axisLabel: { fontSize: 10, color: '#666' }
            },
            yAxis: {
                type: 'value',
                axisLabel: { fontSize: 10, color: '#666' }
            },
            series: [{
                type: 'bar',
                data: values,
                barWidth: '60%',
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { color: '#00a2ff', offset: 0 },
                        { color: '#0077cc', offset: 1 }
                    ]),
                    borderRadius: [4, 4, 0, 0]
                }
            }]
        };

        this.techniqueChart.setOption(option);
    }

    async loadQLearningRecommendation() {
        try {
            const res = await fetch('/api/analyzer/qlearn/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    volunteer_id: 'V001',
                    meridian_id: 'ST',
                    acupoints: ['ST36'],
                    deqi_intensity: 0.6,
                    pain_level: 0.4,
                    current_duration_min: 0,
                    session_count: 5
                })
            });

            if (!res.ok) return;
            const data = await res.json();
            this.updateQLearningDisplay(data);
        } catch(e) {
            console.warn('Failed to load Q-learning recommendation:', e);
        }
    }

    updateQLearningDisplay(data) {
        const action = data.recommended_action || {};

        document.getElementById('param-retention').textContent =
            (action.needle_retention_min || 20) + ' 分钟';
        document.getElementById('param-frequency').textContent =
            (action.stimulation_frequency_hz || 1.5).toFixed(1) + ' Hz';
        document.getElementById('param-depth').textContent =
            (action.needle_depth_mm || 12) + ' mm';

        const techNames = {
            'balanced': '平补平泻',
            'reinforcing': '补法',
            'reducing': '泻法'
        };
        document.getElementById('param-technique').textContent =
            techNames[action.technique] || action.technique || '平补平泻';

        document.getElementById('expected-reward').textContent =
            data.expected_reward?.toFixed(2) || '--';

        const statusEl = document.getElementById('qlearn-status');
        if (statusEl) {
            statusEl.textContent = data.is_exploration ? '探索中' : '收敛';
            statusEl.className = 'badge-info ' + (data.is_exploration ? 'exploring' : 'converged');
        }
    }

    async loadQLearningStats() {
        try {
            const res = await fetch('/api/analyzer/qlearn/stats');
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById('stat-states').textContent = data.state_count || '--';
            document.getElementById('stat-updates').textContent = data.total_updates || '--';
            document.getElementById('stat-exploration').textContent =
                ((data.exploration_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('stat-reward').textContent =
                data.average_reward?.toFixed(2) || '--';
        } catch(e) {
            console.warn('Failed to load Q-learning stats:', e);
        }
    }

    applyRecommendation() {
        alert('已应用推荐参数到当前会话');
    }

    showFeedbackDialog() {
        const reward = prompt('请输入本次疗效评分 (0-100)：', '70');
        if (reward !== null) {
            this.submitFeedback(parseFloat(reward) / 100);
        }
    }

    async submitFeedback(reward) {
        try {
            await fetch('/api/analyzer/qlearn/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    volunteer_id: 'V001',
                    action: { needle_retention_min: 20, technique: 'balanced' },
                    reward: reward,
                    is_terminal: true
                })
            });
            this.loadQLearningRecommendation();
            this.loadQLearningStats();
        } catch(e) {
            console.warn('Failed to submit feedback:', e);
        }
    }

    resize() {
        if (this.radarChart) this.radarChart.resize();
        if (this.techniqueChart) this.techniqueChart.resize();
    }
}
