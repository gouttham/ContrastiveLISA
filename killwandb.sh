ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
