from creditrisk.business_brief import create_business_brief

def test_risk_groups_cover_all_oot_rows(tmp_path):
    report=create_business_brief(n=1200,output_dir=tmp_path)
    groups=report["risk_groups"]
    assert len(groups)==5
    assert [g["group"] for g in groups]==[1,2,3,4,5]
    assert sum(g["applications"] for g in groups)==report["n_out_of_time"]
    assert all(0<=g["average_pd"]<=1 and 0<=g["observed_default_rate"]<=1 for g in groups)
    assert all(a["average_pd"]<=b["average_pd"] for a,b in zip(groups,groups[1:]))
    assert (tmp_path/"business_brief.md").exists()

def test_reproducible_business_report():
    a=create_business_brief(n=1200,seed=42)
    b=create_business_brief(n=1200,seed=42)
    assert a==b
