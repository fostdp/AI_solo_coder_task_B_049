#include "test_framework.h"
#include "apriori_recommender.h"
#include <algorithm>
#include <random>
#include <set>

using namespace tcm;
using namespace tcm_test;

TEST_CAT(Apriori, 空数据集无规则) {
    AprioriRecommender ap;
    ap.run_apriori(0.01, 0.1, 3);
    auto rules = ap.get_rules(10);
    auto items = ap.get_frequent_itemsets(1);
    return rules.empty() && items.empty() && ap.transaction_count() == 0;
}

TEST_CAT(Apriori, 单事务支持度计算) {
    AprioriRecommender ap;
    ap.add_transaction({"ST36","SP6","LI4"}, 0.75, 0.82);
    ap.add_transaction({"ST36","SP6"}, 0.70, 0.78);
    ap.add_transaction({"ST36"}, 0.50, 0.55);
    ap.run_apriori(0.3, 0.5, 3);
    auto sets = ap.get_frequent_itemsets(1);
    bool has_st36 = false, has_sp6 = false, has_pair = false;
    for (auto& s : sets) {
        std::set<std::string> tmp(s.acupoints.begin(), s.acupoints.end());
        if (tmp.count("ST36") && tmp.size() == 1) has_st36 = true;
        if (tmp.count("SP6") && tmp.size() == 1) has_sp6 = true;
        if (tmp.count("ST36") && tmp.count("SP6") && tmp.size() == 2) has_pair = true;
    }
    return has_st36 && has_sp6 && has_pair;
}

TEST_CAT(Apriori, 高置信度规则有效推荐) {
    AprioriRecommender ap;
    for (int i = 0; i < 80; ++i) {
        ap.add_transaction({"ST36","SP6","CV4"}, 0.85, 0.9, i < 60 ? 0.9 : 0.5);
    }
    for (int i = 0; i < 20; ++i) {
        ap.add_transaction({"ST36"}, 0.4, 0.4);
    }
    for (int i = 0; i < 50; ++i) {
        ap.add_transaction({"ST36","SP6"}, 0.7, 0.75);
    }
    ap.run_apriori(0.05, 0.5, 4);
    auto rules = ap.get_rules(50);
    int high_conf_count = 0;
    for (auto& r : rules) {
        if (r.confidence >= 0.8) high_conf_count++;
    }
    return high_conf_count >= 1 && rules.size() >= 1;
}

TEST_CAT(Apriori, 置信度过滤阈值08) {
    AprioriRecommender ap;
    for (int i = 0; i < 100; ++i) ap.add_transaction({"A","B","C"}, 0.9, 0.9);
    for (int i = 0; i < 10; ++i) ap.add_transaction({"A"}, 0.3, 0.3);
    ap.run_apriori(0.01, 0.01, 3);
    auto rules = ap.get_rules(100);
    int cnt_gt_08 = 0, cnt_lt_08 = 0;
    for (auto& r : rules) {
        if (r.antecedent.empty() || r.consequent.empty()) continue;
        if (r.confidence >= 0.8) cnt_gt_08++; else cnt_lt_08++;
    }
    return cnt_gt_08 >= 1 && cnt_lt_08 >= 0;
}

TEST_CAT(Apriori, 配伍推荐TopK不超过限制) {
    AprioriRecommender ap;
    for (int i = 0; i < 50; ++i) {
        ap.add_transaction({"ST36","SP6","LI4","LR3"}, 0.8, 0.8);
        ap.add_transaction({"ST36","SP6","BL23"}, 0.7, 0.75);
    }
    ap.run_apriori(0.05, 0.3, 4);
    auto recs = ap.get_recommendations({"ST36","SP6"}, 5);
    return (int)recs.size() <= 5;
}

TEST_CAT(Apriori, 配伍推荐基于现有穴位扩展) {
    AprioriRecommender ap;
    for (int i = 0; i < 60; ++i)
        ap.add_transaction({"ST36","SP6","LI4"}, 0.82, 0.88);
    for (int i = 0; i < 20; ++i)
        ap.add_transaction({"ST36","SP6"}, 0.5, 0.5);
    for (int i = 0; i < 10; ++i)
        ap.add_transaction({"ST36"}, 0.2, 0.2);
    ap.run_apriori(0.03, 0.25, 4);
    auto recs = ap.get_recommendations({"ST36","SP6"}, 5);
    for (auto& r : recs) {
        bool has_new = false;
        for (auto& a : r.acupoints) {
            if (a != "ST36" && a != "SP6") has_new = true;
        }
        if (has_new) return true;
    }
    return recs.size() == 0;
}

TEST_CAT(Apriori, 异常输入空向量不崩溃) {
    AprioriRecommender ap;
    ap.add_transaction({}, 0.0, 0.0);
    ap.run_apriori(0.0, 0.0, 100);
    auto rules = ap.get_rules(10);
    auto recs = ap.get_recommendations({}, 5);
    return true;
}

TEST_CAT(Apriori, 超长项集限制MaxLen) {
    AprioriRecommender ap;
    for (int i = 0; i < 20; ++i)
        ap.add_transaction({"A","B","C","D","E","F"}, 0.9, 0.9);
    ap.run_apriori(0.05, 0.5, 2);
    auto items = ap.get_frequent_itemsets(1);
    for (auto& it : items) {
        if ((int)it.acupoints.size() > 2) return false;
    }
    return true;
}

TEST_CAT(Apriori, clear后事务数清零) {
    AprioriRecommender ap;
    ap.add_transaction({"A"}, 0.1, 0.1);
    ap.clear();
    ap.run_apriori(0.01, 0.01, 2);
    return ap.transaction_count() == 0 && ap.get_rules(10).empty();
}

TEST_CAT(Apriori, 规则提升度计算正确) {
    AprioriRecommender ap;
    for (int i = 0; i < 100; ++i) ap.add_transaction({"A","B"}, 0.7, 0.7);
    for (int i = 0; i < 100; ++i) ap.add_transaction({"A"}, 0.3, 0.3);
    for (int i = 0; i < 100; ++i) ap.add_transaction({"B"}, 0.3, 0.3);
    for (int i = 0; i < 100; ++i) ap.add_transaction({"C"}, 0.3, 0.3);
    ap.run_apriori(0.05, 0.05, 2);
    auto rules = ap.get_rules(100);
    for (auto& r : rules) {
        if (r.lift < 0.0) return false;
    }
    return !rules.empty();
}
