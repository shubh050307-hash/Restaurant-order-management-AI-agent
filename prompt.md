I am building a restaurant order management ai agent system using langgraph.

I will explain you clearly what i want and what is architecture. I will also explain you the nodes the state and the edges.
In the end I will give you the test cases to simulate and to tell whether they pass or not.

The rough plan is:
There will be a user. when the code runs the user wwill give an order. This will be taken as an input which I will type.
This order should go to the llm. The order as of now should contain the dish and the required quantity.
for simplicity purpose for now we will limit the order to one particular dish and whatever quantity the user wants.
The llm will have to extract the order name and the quantity from the user input.
if the user input is irrelevant to food ordering, the llm should not process that and tell user the same thing that this is an ai agent of food ordering and not a general purpose llm.

Once the llm has order dish and quantity
It will send that to a node called order_confirm
the task of this order confirm is to look at the menu(I will tell you the menu later in this prompt).
then this node will decide one of three cases
the order is available
the order is partially available means quantity is not sufficient
the order is not available at all(dish being not in the menuor 0 quantity available)
It will put this in the status of the state(I will tell you the exact content of the langgraph state as well)

Once the llm received this
if the status is confirm meaning (full available) it should call another node called cook.
If the status is partial or not available it should again prompt the user to decide
the user can either place a new order or can confirm if he wants to go ahead with the partial order

This order retries will bee limited to three attempts meaning if after 3 attempts the user is not satisfied the system  will come to the end node

Now when the cook node is called 
there can be 3 casess
either the cook is done then the status will be ready
and if the cook fails(we can use a probability function). Give 40% chances of fail and 60% chances of success

if the cook fails there shoukd be 1 more attempt allowed to cook to succedd. if the cook fails even after these then llm should issue a apology to the user and come to END state

if the cook succedd s, the status will be READY 
and the next node will be called wwhich is serve
similar to cook this also has 2 cases
serve pass or serve fails
this also has 2 retry attempts
if the serve fails 2 times then the llm should issue a apology to the user and come to END state

if the serve succedd then the status should become complete and the llm issue a message to the user saying your order is completed
if the serve fails then cook should be called one more time to retry
Note that if cook has exhausted its retry attempts then it should not cook again and llm should issue a apology and come to END state

Now the state of langgraph

there should be a annotated message between the llm and user

there should be order details
dish name as string
required quantity as integer
order_confirm will write the available quantity by reading the menu
the llm should get to know the order confirm status by reading the state
if a dish is not available in the menu then the order-confirm should write 0 as available quantity

then there sould be status 
each node will update the status as specified in the above rules
then order retry attempts which are 3
cook retry attempts which are 2
serve retry attempts which are 2
each time a failure happened and a node is retrying it should decrement the counter
if any retry counter becomes 0 it means it is over. llm should understand whether it has to give retry or issue apology by reading this counter

In the end there should be final result whether the order is completed or not

this is the menu
burger 10
pizza 20
momos 4

Write the code and ask if there are any open questions from your side then ask
I will give you some test scenarios to test later on

Use llm as Groq llm and the name of model is openai/gpt-oss-120b
It's api key is in .env file

TC1
a user asks unrelated question
then he places a order which is partial
then he rejects the partial and wants the order again
again he orders a dish which is not available
expected. END due to order retry

TC2
user places a order which can be available
cook makes it fails again
cook again suceess
serve fails once
cook retries success
serve does success
overall success

TC3
user order partial 
does not want partial
orders again fully available
cook makes it fails
makes again success
serve fails
cook retries success
serve fails again
cook doesnot retry due to retry exhaust
overall FAIL
