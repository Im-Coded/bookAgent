from ai_agent.agent import AIAgent
import time

def main():
    agent = AIAgent()
    agent.schedule_tasks()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 