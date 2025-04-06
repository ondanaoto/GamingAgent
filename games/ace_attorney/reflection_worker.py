from collections import deque

class ReflectionTracker:
    def __init__(self):
        self.lives = 5
        self.success = 0
        self.dialog_log = []
        self.pending_presentations = deque()
        self.last_r_index = -1  # index in dialog_log when 'r' was last pressed

    def log_dialog(self, dialog):
        if dialog and dialog["name"] and dialog["text"]:
            self.dialog_log.append(dialog)

    def record_r_press(self):
        self.last_r_index = len(self.dialog_log)

    def record_x_press(self, evidence):
        if self.last_r_index >= 0:
            self.pending_presentations.append({
                "start_idx": self.last_r_index,
                "evidence": evidence.get("name", "UNKNOWN")
            })
            self.last_r_index = -1

    def check_pending(self):
        results = []
        while self.pending_presentations:
            entry = self.pending_presentations.popleft()
            future_dialogs = self.dialog_log[entry["start_idx"] + 1 : entry["start_idx"] + 6]
            rejected = any(
                "they aren't, are they" in d['text'].lower() and "phoenix" in d['name'].lower()
                for d in future_dialogs
            )
            if rejected:
                self.lives -= 1
                print(f"[REFLECTION] ❌ Judge rejected evidence: {entry['evidence']}")
            else:
                self.success += 1
                print(f"[REFLECTION] ✅ Evidence accepted: {entry['evidence']}")
            results.append(not rejected)
        return results

    def print_progress(self):
        success_bar = "#" * self.success + "." * (10 - self.success)
        life_bar = "♥" * (5 - self.lives) + "." * self.lives
        print(f"\nProgress: [{success_bar}] {self.success}/10")
        print(f"Lives:    [{life_bar}] {self.lives}/5")
        print("=" * 70)
