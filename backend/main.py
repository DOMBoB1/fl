from gui_app import ClassroomMonitorGUI, MonitorStats
from engine import MonitoringEngine


def main():
    engine = MonitoringEngine()

    def frame_provider():
        return engine.get_frame()

    def stats_provider():
        return MonitorStats(**engine.get_stats())

    gui = ClassroomMonitorGUI(
        frame_provider=frame_provider,
        stats_provider=stats_provider,
        start_cb=engine.start,
        stop_cb=engine.stop,
        title="Classroom Monitor",
        target_fps=15
    )

    gui.run()


if __name__ == "__main__":
    main()
