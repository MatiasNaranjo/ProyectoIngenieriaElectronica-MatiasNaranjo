import statistics
import time


def warmup(model, frame, n, **predict_kwargs):
    """Ejecuta n predicciones descartadas para estabilizar cachés / lazy init."""
    for _ in range(n):
        model.predict(frame, **predict_kwargs)


def run_benchmark(model, frame, num_frames, **predict_kwargs):
    """
    Cronometra num_frames llamadas a model.predict() sobre un frame fijo.

    Devuelve una lista de tiempos wall-clock en milisegundos.
    """
    times = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        model.predict(frame, **predict_kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def summary_stats(times):
    """
    Estadísticas resumidas de una lista de tiempos (ms).

    Devuelve dict con n, mean, std, min, p50, p95, p99, max.
    """
    q = statistics.quantiles(times, n=100)
    return {
        "n": len(times),
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times),
        "min_ms": min(times),
        "p50_ms": q[49],
        "p95_ms": q[94],
        "p99_ms": q[98],
        "max_ms": max(times),
    }
