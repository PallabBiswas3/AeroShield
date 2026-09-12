"""Deterministic, reviewable public-health advisory templates."""
from __future__ import annotations


MESSAGES = {
    "EN": {
        "LOW": {
            "general": "Normal outdoor activity is generally appropriate.",
            "sensitive": "Sensitive individuals should monitor symptoms and local updates.",
            "schools": "Normal school activities may continue with routine monitoring.",
            "outdoor_workers": "Normal work may continue with access to breaks and hydration.",
        },
        "ELEVATED": {
            "general": "Reduce prolonged or strenuous outdoor activity during the indicated risk window.",
            "sensitive": "Children, older adults, and people with heart or lung conditions should reduce outdoor exposure and follow their clinician's advice.",
            "schools": "Move prolonged outdoor school activities indoors where practical.",
            "outdoor_workers": "Use work-rest cycles, reduce strenuous exposure, and follow employer respiratory-protection procedures.",
        },
        "SEVERE": {
            "general": "Avoid prolonged outdoor exposure during the indicated risk window and follow official local advisories.",
            "sensitive": "Remain indoors where possible. Seek medical help for severe breathing difficulty or chest pain.",
            "schools": "Suspend prolonged outdoor school activities and keep vulnerable students under observation.",
            "outdoor_workers": "Postpone strenuous work where possible and apply the site's approved exposure-control plan.",
        },
    },
    "HI": {
        "LOW": {
            "general": "सामान्य बाहरी गतिविधियां आम तौर पर की जा सकती हैं।",
            "sensitive": "संवेदनशील व्यक्ति लक्षणों और स्थानीय सूचनाओं पर ध्यान रखें।",
            "schools": "नियमित निगरानी के साथ स्कूल की सामान्य गतिविधियां जारी रह सकती हैं।",
            "outdoor_workers": "विश्राम और पानी की सुविधा के साथ सामान्य काम जारी रह सकता है।",
        },
        "ELEVATED": {
            "general": "दिए गए जोखिम समय में लंबे समय तक या बहुत मेहनत वाली बाहरी गतिविधि कम करें।",
            "sensitive": "बच्चे, बुजुर्ग और हृदय या फेफड़ों की बीमारी वाले लोग बाहर का संपर्क कम करें और डॉक्टर की सलाह मानें।",
            "schools": "जहां संभव हो, लंबे समय वाली बाहरी स्कूल गतिविधियों को अंदर कराएं।",
            "outdoor_workers": "काम और विश्राम का चक्र अपनाएं तथा स्वीकृत श्वसन-सुरक्षा प्रक्रिया का पालन करें।",
        },
        "SEVERE": {
            "general": "दिए गए जोखिम समय में लंबे बाहरी संपर्क से बचें और आधिकारिक स्थानीय सलाह का पालन करें।",
            "sensitive": "जहां संभव हो घर के अंदर रहें। सांस लेने में गंभीर कठिनाई या सीने में दर्द होने पर चिकित्सा सहायता लें।",
            "schools": "लंबी बाहरी स्कूल गतिविधियां रोकें और संवेदनशील विद्यार्थियों की निगरानी करें।",
            "outdoor_workers": "जहां संभव हो भारी काम टालें और कार्यस्थल की स्वीकृत जोखिम-नियंत्रण योजना लागू करें।",
        },
    },
}


def advisory_level(pm25_upper: float) -> str:
    if pm25_upper >= 120:
        return "SEVERE"
    if pm25_upper >= 60:
        return "ELEVATED"
    return "LOW"


def generate_health_advisory(pm25_lower: float, pm25_upper: float, ward_name: str = "Selected Delhi grid", languages=("EN", "HI")) -> dict:
    if pm25_lower < 0 or pm25_upper < pm25_lower:
        raise ValueError("PM2.5 interval must be non-negative and ordered")
    level = advisory_level(pm25_upper)
    rendered = {}
    for language in languages:
        code = language.upper()
        if code not in MESSAGES:
            raise ValueError(f"unsupported advisory language: {language}")
        rendered[code] = MESSAGES[code][level]
    return {
        "ward": ward_name,
        "risk_level": level,
        "basis": {"pm25_lower": pm25_lower, "pm25_upper": pm25_upper, "decision_value": "upper_interval_bound"},
        "languages": rendered,
        "medical_disclaimer": "General risk communication only; not individual medical advice. Follow official authorities and healthcare professionals.",
        "template_version": "2026-09-12.v1",
    }
