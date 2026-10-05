import React, { useEffect, useMemo, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Config                                                             */
/* ------------------------------------------------------------------ */

const API_URL = "https://room-type-predictor-8hbw.onrender.com/predict";

// Order must match your model's `classes_` (scikit-learn sorts them alphabetically).
// Change these if your label encoder produced a different order.
const CLASS_LABELS = ["Entire home/apt", "Private room", "Shared room"];

// Accent per room type: dot/bar color + soft badge background.
const CLASS_STYLES = {
  "Entire home/apt": {
    bar: "bg-teal-500 dark:bg-teal-400",
    badge:
      "bg-teal-500/15 text-teal-800 border-teal-600/30 dark:bg-teal-400/15 dark:text-teal-200 dark:border-teal-300/30",
    dot: "bg-teal-500 dark:bg-teal-300",
  },
  "Private room": {
    bar: "bg-amber-500 dark:bg-amber-400",
    badge:
      "bg-amber-500/15 text-amber-900 border-amber-600/30 dark:bg-amber-400/15 dark:text-amber-200 dark:border-amber-300/30",
    dot: "bg-amber-500 dark:bg-amber-300",
  },
  "Shared room": {
    bar: "bg-rose-500 dark:bg-rose-400",
    badge:
      "bg-rose-500/15 text-rose-800 border-rose-600/30 dark:bg-rose-400/15 dark:text-rose-200 dark:border-rose-300/30",
    dot: "bg-rose-500 dark:bg-rose-300",
  },
};
const FALLBACK_STYLE = {
  bar: "bg-indigo-500 dark:bg-indigo-400",
  badge:
    "bg-indigo-500/15 text-indigo-800 border-indigo-600/30 dark:bg-indigo-400/15 dark:text-indigo-200 dark:border-indigo-300/30",
  dot: "bg-indigo-500 dark:bg-indigo-300",
};

const NEIGHBOURHOODS = {
  Manhattan: [
    "Midtown",
    "Harlem",
    "Upper West Side",
    "Upper East Side",
    "Hell's Kitchen",
    "East Village",
    "Chelsea",
    "Lower East Side",
    "Washington Heights",
    "Financial District",
  ],
  Brooklyn: [
    "Williamsburg",
    "Bedford-Stuyvesant",
    "Bushwick",
    "Crown Heights",
    "Greenpoint",
    "Park Slope",
    "Flatbush",
    "Sunset Park",
    "Fort Greene",
    "Prospect-Lefferts Gardens",
  ],
  Queens: [
    "Astoria",
    "Long Island City",
    "Flushing",
    "Ridgewood",
    "Sunnyside",
    "Jamaica",
    "Woodside",
    "Forest Hills",
  ],
  Bronx: ["Kingsbridge", "Fordham", "Mott Haven", "Concourse", "Riverdale"],
  "Staten Island": ["St. George", "Tompkinsville", "Stapleton", "Great Kills"],
};

// Numeric fields: drives rendering and validation (mirrors the Pydantic model).
const NUMERIC_FIELDS = {
  latitude: { label: "Latitude", kind: "float", min: -90, max: 90, hint: "-90 to 90" },
  longitude: { label: "Longitude", kind: "float", min: -180, max: 180, hint: "-180 to 180" },
  price: { label: "Price per night (USD)", kind: "float", min: 0, exclusiveMin: true, hint: "Greater than 0" },
  minimum_nights: { label: "Minimum nights", kind: "int", min: 1, max: 365, hint: "1 to 365" },
  availability_365: { label: "Days available per year", kind: "int", min: 0, max: 365, hint: "0 to 365" },
  number_of_reviews: { label: "Total reviews", kind: "int", min: 0, hint: "0 or more" },
  reviews_per_month: { label: "Reviews per month", kind: "float", min: 0, hint: "0 or more" },
  calculated_host_listings_count: { label: "Host's listings", kind: "int", min: 0, hint: "0 or more" },
};

const INITIAL_FORM = {
  neighbourhood_group: "Manhattan",
  neighbourhood: "Midtown",
  latitude: "40.7549",
  longitude: "-73.984",
  price: "150",
  minimum_nights: "3",
  availability_365: "180",
  number_of_reviews: "24",
  reviews_per_month: "0.8",
  calculated_host_listings_count: "1",
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function validateField(name, raw) {
  const value = String(raw ?? "").trim();
  if (name === "neighbourhood_group" || name === "neighbourhood") {
    return value ? "" : "Required";
  }
  const f = NUMERIC_FIELDS[name];
  if (!f) return "";
  if (value === "") return "Required";
  const n = Number(value);
  if (!Number.isFinite(n)) return "Enter a number";
  if (f.kind === "int" && !Number.isInteger(n)) return "Use a whole number";
  if (f.exclusiveMin && n <= f.min) return `Must be greater than ${f.min}`;
  if (!f.exclusiveMin && f.min !== undefined && n < f.min) return `Must be ${f.min} or more`;
  if (f.max !== undefined && n > f.max) return `Must be ${f.max} or less`;
  return "";
}

function validateAll(form) {
  const errors = {};
  Object.keys(form).forEach((k) => {
    const msg = validateField(k, form[k]);
    if (msg) errors[k] = msg;
  });
  return errors;
}

function buildPayload(form) {
  return {
    latitude: parseFloat(form.latitude),
    longitude: parseFloat(form.longitude),
    price: parseFloat(form.price),
    minimum_nights: parseInt(form.minimum_nights, 10),
    number_of_reviews: parseInt(form.number_of_reviews, 10),
    reviews_per_month: parseFloat(form.reviews_per_month),
    calculated_host_listings_count: parseInt(form.calculated_host_listings_count, 10),
    availability_365: parseInt(form.availability_365, 10),
    neighbourhood_group: form.neighbourhood_group.trim(),
    neighbourhood: form.neighbourhood.trim(),
  };
}

// Turns FastAPI's 422 detail array (or any other error body) into readable text.
function describeApiError(status, body) {
  if (body && Array.isArray(body.detail)) {
    return body.detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join("\n");
  }
  if (body && typeof body.detail === "string") return body.detail;
  return `The server responded with status ${status}.`;
}

/* ------------------------------------------------------------------ */
/*  Small UI pieces                                                    */
/* ------------------------------------------------------------------ */

const glass =
  "rounded-3xl border border-white/50 bg-white/40 backdrop-blur-xl shadow-[0_10px_40px_rgba(30,41,59,0.12)] " +
  "dark:border-white/10 dark:bg-slate-900/40 dark:shadow-[0_10px_40px_rgba(0,0,0,0.5)]";

const inputBase =
  "w-full rounded-xl border bg-white/50 px-3.5 py-2.5 text-sm text-slate-900 placeholder-slate-400 " +
  "outline-none transition focus:ring-2 focus:ring-teal-500/60 motion-reduce:transition-none " +
  "dark:bg-slate-950/40 dark:text-slate-100 dark:placeholder-slate-500 dark:focus:ring-teal-300/60";

function Field({ id, label, hint, error, children }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="flex items-baseline justify-between gap-2 text-sm font-medium text-slate-800 dark:text-slate-200">
        <span>{label}</span>
        {hint && <span className="text-xs font-normal text-slate-500 dark:text-slate-400">{hint}</span>}
      </label>
      {children}
      <p
        id={`${id}-error`}
        role={error ? "alert" : undefined}
        className={`min-h-[1rem] text-xs text-rose-600 dark:text-rose-300 ${error ? "" : "invisible"}`}
      >
        {error || "."}
      </p>
    </div>
  );
}

function Spinner({ className = "h-5 w-5" }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function ThemeToggle({ dark, onToggle }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label="Dark mode"
      onClick={onToggle}
      className="relative inline-flex h-10 w-[4.5rem] shrink-0 items-center rounded-full border border-white/50 bg-white/40 p-1 backdrop-blur-xl
                 shadow-inner transition-colors duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500
                 dark:border-white/10 dark:bg-slate-900/50 motion-reduce:transition-none"
    >
      <span
        className={`flex h-8 w-8 items-center justify-center rounded-full bg-white text-amber-500 shadow-md transition-transform duration-300
                    dark:bg-slate-700 dark:text-indigo-200 motion-reduce:transition-none ${dark ? "translate-x-8" : "translate-x-0"}`}
      >
        {dark ? (
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            <path d="M21 14.5A9 9 0 0 1 9.5 3a.75.75 0 0 0-1-.9A10.5 10.5 0 1 0 21.9 15.5a.75.75 0 0 0-.9-1Z" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <circle cx="12" cy="12" r="4" fill="currentColor" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </svg>
        )}
      </span>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/*  Result panel                                                       */
/* ------------------------------------------------------------------ */

function ResultPanel({ status, result, error }) {
  const [grown, setGrown] = useState(false);

  // Bars start at 0% and grow once the result mounts.
  useEffect(() => {
    if (status !== "success") {
      setGrown(false);
      return undefined;
    }
    const id = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(id);
  }, [status, result]);

  if (status === "loading") {
    return (
      <div className="flex min-h-[22rem] flex-col items-center justify-center gap-4 text-center" aria-live="polite">
        <Spinner className="h-10 w-10 text-teal-600 dark:text-teal-300" />
        <div>
          <p className="font-display text-xl font-semibold">Running the model</p>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">This usually takes a moment.</p>
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex min-h-[22rem] flex-col justify-center gap-3" role="alert">
        <div className="flex h-11 w-11 items-center justify-center rounded-full bg-rose-500/15 text-rose-600 dark:text-rose-300">
          <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M12 8v5M12 16.5v.01" />
            <path d="M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
          </svg>
        </div>
        <h2 className="font-display text-xl font-semibold">Prediction failed</h2>
        <p className="whitespace-pre-line text-sm leading-relaxed text-slate-700 dark:text-slate-300">{error}</p>
      </div>
    );
  }

  if (status === "success" && result) {
    const style = CLASS_STYLES[result.label] || FALLBACK_STYLE;
    return (
      <div className="flex flex-col gap-6" aria-live="polite">
        <div>
          <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${style.badge}`}>
            <span className={`h-2 w-2 rounded-full ${style.dot}`} />
            Predicted room type
          </span>
          <h2 className="font-display mt-4 text-4xl font-bold leading-tight tracking-tight sm:text-5xl">{result.label}</h2>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
            {(result.top * 100).toFixed(1)}% confidence
          </p>
        </div>

        <ul className="flex flex-col gap-4">
          {result.rows.map((row) => {
            const s = CLASS_STYLES[row.label] || FALLBACK_STYLE;
            const isTop = row.label === result.label;
            return (
              <li key={row.label}>
                <div className="mb-1.5 flex items-baseline justify-between text-sm">
                  <span className={isTop ? "font-semibold" : "text-slate-700 dark:text-slate-300"}>{row.label}</span>
                  <span className="tabular-nums text-slate-600 dark:text-slate-400">{(row.value * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-900/10 dark:bg-white/10" role="presentation">
                  <div
                    className={`h-full rounded-full transition-[width] duration-700 ease-out motion-reduce:transition-none ${s.bar} ${isTop ? "" : "opacity-60"}`}
                    style={{ width: grown ? `${Math.max(row.value * 100, 0.5)}%` : "0%" }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    );
  }

  return (
    <div className="flex min-h-[22rem] flex-col justify-center gap-3">
      <div className="flex h-11 w-11 items-center justify-center rounded-full bg-teal-500/15 text-teal-700 dark:text-teal-300">
        <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 11.5 12 4l9 7.5" />
          <path d="M5 10v9h14v-9" />
          <path d="M10 19v-5h4v5" />
        </svg>
      </div>
      <h2 className="font-display text-xl font-semibold">No prediction yet</h2>
      <p className="max-w-sm text-sm leading-relaxed text-slate-600 dark:text-slate-400">
        Enter the listing details and select Predict room type. The model's verdict and the probability for each room type will appear here.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */

export default function App() {
  const [dark, setDark] = useState(() => {
    try {
      const saved = localStorage.getItem("room-predictor-theme");
      if (saved) return saved === "dark";
    } catch (_) {
      /* storage unavailable */
    }
    return typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : false;
  });
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    try {
      localStorage.setItem("room-predictor-theme", dark ? "dark" : "light");
    } catch (_) {
      /* storage unavailable */
    }
  }, [dark]);

  const suggestions = useMemo(() => NEIGHBOURHOODS[form.neighbourhood_group] || [], [form.neighbourhood_group]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => {
      const next = { ...prev, [name]: value };
      if (name === "neighbourhood_group") {
        next.neighbourhood = (NEIGHBOURHOODS[value] || [])[0] || "";
      }
      return next;
    });
    if (errors[name] || (name === "neighbourhood_group" && errors.neighbourhood)) {
      setErrors((prev) => ({ ...prev, [name]: validateField(name, value), ...(name === "neighbourhood_group" ? { neighbourhood: "" } : {}) }));
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setErrors((prev) => ({ ...prev, [name]: validateField(name, value) }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const found = validateAll(form);
    setErrors(found);
    if (Object.keys(found).length > 0) {
      const first = Object.keys(found)[0];
      const el = document.getElementById(first);
      if (el) el.focus();
      return;
    }

    setStatus("loading");
    setApiError("");
    setResult(null);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload(form)),
        signal: controller.signal,
      });

      let body = null;
      try {
        body = await res.json();
      } catch (_) {
        /* non-JSON body */
      }

      if (!res.ok) {
        throw new Error(describeApiError(res.status, body));
      }
      if (!body || typeof body.Predicted_room_type !== "string") {
        throw new Error("The API replied, but the response was missing Predicted_room_type.");
      }

      // predict_proba often returns [[...]]; accept both shapes.
      const raw = Array.isArray(body.Probability)
        ? Array.isArray(body.Probability[0])
          ? body.Probability[0]
          : body.Probability
        : [];
      const rows = raw.map((value, i) => ({
        label: CLASS_LABELS[i] || `Class ${i + 1}`,
        value: Number(value) || 0,
      }));
      const predicted = body.Predicted_room_type;
      const top = (rows.find((r) => r.label === predicted) || rows.reduce((a, b) => (b.value > (a?.value ?? -1) ? b : a), null))?.value ?? 0;

      setResult({ label: predicted, rows, top });
      setStatus("success");
    } catch (err) {
      if (err.name === "AbortError") {
        setApiError("The request timed out after 20 seconds. Check that the FastAPI server is running and responding.");
      } else if (err instanceof TypeError) {
        setApiError(
          `Can't reach the API at ${API_URL}.\nStart the FastAPI server and make sure CORS allows this origin.`
        );
      } else {
        setApiError(err.message || "Something went wrong while requesting a prediction.");
      }
      setStatus("error");
    } finally {
      clearTimeout(timeout);
    }
  };

  const textInput = (name, extra = {}) => (
    <input
      id={name}
      name={name}
      value={form[name]}
      onChange={handleChange}
      onBlur={handleBlur}
      aria-invalid={Boolean(errors[name])}
      aria-describedby={`${name}-error`}
      className={`${inputBase} ${errors[name] ? "border-rose-500/70" : "border-white/60 dark:border-white/10"}`}
      {...extra}
    />
  );

  const numberField = (name) => {
    const f = NUMERIC_FIELDS[name];
    return (
      <Field key={name} id={name} label={f.label} hint={f.hint} error={errors[name]}>
        {textInput(name, {
          type: "number",
          inputMode: f.kind === "int" ? "numeric" : "decimal",
          step: f.kind === "int" ? "1" : "any",
          min: f.exclusiveMin ? undefined : f.min,
          max: f.max,
        })}
      </Field>
    );
  };

  return (
    <div className={dark ? "dark" : ""}>
      {/* Fonts + keyframes kept inside this file so App.js stays self-contained */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap');
        .font-display { font-family: 'Bricolage Grotesque', ui-sans-serif, system-ui, sans-serif; letter-spacing: -0.02em; }
        .font-body { font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif; }
        @keyframes drift-a { 0%,100% { transform: translate3d(0,0,0) } 50% { transform: translate3d(40px,30px,0) } }
        @keyframes drift-b { 0%,100% { transform: translate3d(0,0,0) } 50% { transform: translate3d(-50px,-20px,0) } }
        .orb-a { animation: drift-a 26s ease-in-out infinite; }
        .orb-b { animation: drift-b 32s ease-in-out infinite; }
        @media (prefers-reduced-motion: reduce) { .orb-a, .orb-b { animation: none; } }
        input[type=number]::-webkit-inner-spin-button { opacity: .5; }
      `}</style>

      <div className="font-body relative min-h-screen overflow-hidden bg-gradient-to-br from-sky-100 via-amber-50 to-teal-100 text-slate-900 transition-colors duration-500 dark:from-slate-950 dark:via-indigo-950 dark:to-slate-900 dark:text-slate-100 motion-reduce:transition-none">
        {/* Backdrop: blurred light orbs + a faint street grid so the glass has something to refract */}
        <div className="pointer-events-none absolute inset-0" aria-hidden="true">
          <div className="orb-a absolute -left-24 -top-24 h-96 w-96 rounded-full bg-teal-400/50 blur-3xl dark:bg-teal-500/30" />
          <div className="orb-b absolute right-[-6rem] top-1/3 h-[28rem] w-[28rem] rounded-full bg-amber-300/60 blur-3xl dark:bg-amber-500/20" />
          <div className="orb-a absolute bottom-[-8rem] left-1/3 h-96 w-96 rounded-full bg-rose-300/50 blur-3xl dark:bg-indigo-500/30" />
          <div
            className="absolute inset-0 opacity-[0.07] dark:opacity-[0.09]"
            style={{
              backgroundImage:
                "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)",
              backgroundSize: "56px 56px",
            }}
          />
        </div>

        <main className="relative z-10 mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:py-12">
          <header className="mb-8 flex items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-3xl font-bold sm:text-4xl">Room type predictor</h1>
              <p className="mt-2 max-w-xl text-sm leading-relaxed text-slate-700 dark:text-slate-300 sm:text-base">
                Describe an Airbnb listing and the model estimates whether it is an entire home, a private room, or a shared room.
              </p>
            </div>
            <ThemeToggle dark={dark} onToggle={() => setDark((d) => !d)} />
          </header>

          <div className="grid items-start gap-6 lg:grid-cols-5">
            {/* Form */}
            <form onSubmit={handleSubmit} noValidate className={`${glass} p-5 sm:p-7 lg:col-span-3`}>
              <fieldset className="mb-2">
                <legend className="font-display mb-3 text-lg font-semibold">Where it is</legend>
                <div className="grid gap-x-4 sm:grid-cols-2">
                  <Field id="neighbourhood_group" label="Borough" error={errors.neighbourhood_group}>
                    <select
                      id="neighbourhood_group"
                      name="neighbourhood_group"
                      value={form.neighbourhood_group}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      className={`${inputBase} border-white/60 dark:border-white/10`}
                    >
                      {Object.keys(NEIGHBOURHOODS).map((g) => (
                        <option key={g} value={g} className="text-slate-900">
                          {g}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field id="neighbourhood" label="Neighbourhood" hint="Pick or type" error={errors.neighbourhood}>
                    {textInput("neighbourhood", { type: "text", list: "neighbourhood-options", autoComplete: "off" })}
                    <datalist id="neighbourhood-options">
                      {suggestions.map((n) => (
                        <option key={n} value={n} />
                      ))}
                    </datalist>
                  </Field>
                  {numberField("latitude")}
                  {numberField("longitude")}
                </div>
              </fieldset>

              <fieldset className="mb-2 mt-4">
                <legend className="font-display mb-3 text-lg font-semibold">Price and availability</legend>
                <div className="grid gap-x-4 sm:grid-cols-3">
                  {numberField("price")}
                  {numberField("minimum_nights")}
                  {numberField("availability_365")}
                </div>
              </fieldset>

              <fieldset className="mt-4">
                <legend className="font-display mb-3 text-lg font-semibold">Reviews and host</legend>
                <div className="grid gap-x-4 sm:grid-cols-3">
                  {numberField("number_of_reviews")}
                  {numberField("reviews_per_month")}
                  {numberField("calculated_host_listings_count")}
                </div>
              </fieldset>

              <button
                type="submit"
                disabled={status === "loading"}
                className="mt-4 inline-flex w-full items-center justify-center gap-2.5 rounded-2xl bg-slate-900 px-6 py-3.5 text-base font-semibold text-white shadow-lg
                           transition hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2
                           disabled:cursor-not-allowed disabled:opacity-70 dark:bg-teal-400 dark:text-slate-950 dark:hover:bg-teal-300
                           dark:focus-visible:ring-offset-slate-900 motion-reduce:transition-none"
              >
                {status === "loading" ? (
                  <>
                    <Spinner />
                    Predicting
                  </>
                ) : (
                  "Predict room type"
                )}
              </button>
            </form>

            {/* Result */}
            <section className={`${glass} p-5 sm:p-7 lg:sticky lg:top-8 lg:col-span-2`} aria-label="Prediction result">
              <ResultPanel status={status} result={result} error={apiError} />
            </section>
          </div>

          <footer className="mt-8 text-center text-xs text-slate-600 dark:text-slate-400">
            Sending requests to <code className="rounded bg-slate-900/10 px-1.5 py-0.5 dark:bg-white/10">{API_URL}</code>
          </footer>
        </main>
      </div>
    </div>
  );
}
