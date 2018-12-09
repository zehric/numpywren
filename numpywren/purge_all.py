import boto3
def sqs_purge_queue(sqs_queue_name):
    sqs = boto3.resource('sqs', region_name='us-west-2')
    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=sqs_queue_name)
    queue.purge()
def purge_all_queues():
    for i in range(128):
        sqs_purge_queue(gen_instance_unique_name('pywren-jobs-1', i))

def gen_instance_unique_name(base, instance_id):
    return "{0}_{1}".format(base, instance_id)

purge_all_queues()
