# TELECI-APP
Data visualisation web application. [DEMO](http://teleci.herokuapp.com/)

## Application Architecture
### Data processing model
Transformation to xAPI and extraction
This step works similarly to ETL data management approaches. Raw streams of data from virtual infrastructure, software, and applications are ingested either in their entirety or according to predefined rules. Virtually every LMS has an internal database that is collecting user activity and system data more often called a "log". For our experimental testbed we used Moodle. This LMS has many different data logging possibilities. Logs are available at site and course level:

* A log of activity in the course may be generated in reports.
* You can filter the logs by level -
  * teacher level - an event or action performed by a teacher (usually) which affects the students' learning experience. This might be for instance, grading a student or adding a module to the course,
  * participating level - an event or action which could be related to a user's learning experience. This might be for instance a student posting to a forum or submitting an assignment. You can see what pages the student accessed, the time and date they accessed it, the IP address they came from, and their actions (view, add, update, delete). You can choose to display the logs on a page or download them in text, ODS or Excel format.
* A log of site activity may be generated by a site administrator. The log can display all activities, site news or site errors, such as failed login attempts, and all levels.
* Live logs are also available for each course
Careful selection of a few these logging features were incorporated in custom data export that was later adapted to xAPI specifications. The possibilities of xAPI are enormous, but its implementation and execution across different authoring solutions can vary greatly. Because xAPI is an open standard, tools that enable “JavaScript tweaking” can provided us with an avenue by which we could realize some of the possibilities. However, coding is needed. However we also tried to  focus on some of the many common tools which are available to everyone, in terms of xAPI support, right out of the box. xAPI solutions harness the power of new technologies in order to push improvements, security, and compliance across the enterprise. xAPI also leverages the native capabilities of modern cloud data warehouses and big data processing frameworks. The xAPI process is adaptable and flexible, so it’s suitable for a variety of educational entities, applications, and goals. The scalability of a cloud infrastructure and hosted services like integration platform-as-a-service (iPaaS) and software-as-a-service (SaaS) give organizations the ability to expand resources on the fly. They add the compute time and storage space necessary for even massive data transformation tasks. xAPI appears to be the future of data integration, offering many advantages over SCORM, which is an older, slower process. Data volume has grown exponentially for organizations, and older tools cannot efficiently handle the integration of all this data into a repository for analysis. xAPI delivers better agility and less maintenance, making it a cost-effective way for educational institutions of all sizes to take advantage of cloud-based data processing and visualization. In our research we found that there were four key areas where tools deviated in their approach and support of xAPI:

* xAPI built in statements: Specifically, what statements does the tool send automatically without author intervention, other than creating content.
* Custom xAPI statements: Moving beyond the out-of-the-box statements, what options are there to send custom statements in the same way you might trigger a show-image action with programming.
* Support for JavaScript: Very important since large part of the research was focusing on coding aspect of the specification
* xAPI deployment: While the design for publishing xAPI packages is elegant, the implementation isn’t as simple as SCORM in most solutions. We focused on the limitations, how to stream, and if there was a need to make changes post publish (data cleaning and aggregation).
In our custom deployment we used custom xAPI statements that incorporated the resulting data from TELECI method of content structure. For first iteration the xAPI data stream export was manually accessed using predefined custom options such as date an course (pic). 
![image](https://user-images.githubusercontent.com/7984338/127028933-d84879fe-d158-40a0-8a59-5f0991148740.png)
Pic: First iteration of manual data export (.xlsx file)

Such solution was temporary sufficient to continue developing xAPI data preparation in order to model Telecides but not nearly complete functionality to have it automated. The next step was to develop data pipeline or stream that could provide automatic data extraction on demand. In the project's test-bed Moodle platform Rest xAPI was developed which could be used to further improve the automation process for data visualization in form of Telecides and other relevant visualizations (pic).

![image](https://user-images.githubusercontent.com/7984338/127028960-84296f3c-83bb-4521-bd12-3122408d3e95.png)
Pic: API request (JSON file)

In such solution the access to stream of learning experience API data was pivotal to further develop the web application. Using Python and corresponding libraries such as Pandas we were able to load data from Moodle's API accordingly to xAPI specifications. 

Conceptual and physical data model
xAPI standard is presented as an e-learning standard for tracking data interoperability in the whole learning process. It does not focus particularly on assessment. Our objective is to examine the xAPI data model dedicated to conceive the TELECI method data and show that certain analytics in the assessment process are possible to achieve when using xAPI as is. When we observe the data model of xAPI, we note that the activity statement is composed of a minimum of three properties namely the actor, the verb and the object. All others properties are optional. An activity statement is presented through an example in figure 1 below.

![image](https://user-images.githubusercontent.com/7984338/127029019-48cb81da-6eac-4e9d-8161-72931f3e16f1.png)
Pic: Example of a xAPI learning activity statement.

To express statements related to the assessment activities, each statement is described with an optional property which is the result property storing the assessment results in the context of e-learning. In the image below there is "correct_answer" feature that collects if user answered question correctly (true) or incorrectly (false). 

![image](https://user-images.githubusercontent.com/7984338/127029070-cbdacebb-d5cb-49db-b95a-7385f7e8298c.png)
Pic: A set of xAPI activities send through Rest API that have been aggregated using Pandas library

Initial setup using Dash framework is build in Pycharm IDE using Python and JavaScript. Web application is deployed offline using Python Flask server for development purposes. Application incorporates pre-processing code from Jupyter notebook together with Dash framework for interactive data visualisation applications.

Python Library requirements:
* Dash - core application functionality
* Dash Core Components - defining interactivity
* Dash Bootstrap components - defining layout
* Dash HTML - provides application with HTML capabilities
* Numpy - adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* Pandas - data manipulation and analysis
* Plotly - interactive data visualisations build on top of D3.js. Used to develope Telecides

![image](https://user-images.githubusercontent.com/7984338/127028050-a1f6bac5-d6a6-4290-9ec0-5834c4fb9631.png)
Pic: Process Flow
Data stream that is read from raw JSON format from Moodles API is then cleaned and aggregated to form necessary dataframes for visualising Telecides. By manipulating multidimensional arrays of data (see Apendix) we can create new dataframe that consists of information about each course unit and the question pairs necessary to form Telecides (Table 1).
![image](https://user-images.githubusercontent.com/7984338/127029219-76c206e7-04a0-4a2e-9a61-9f2868858463.png)
Table 1: Sum of all question pairs for each course unit ("section")

This dataframe is further evolved to define dataframe with average probability for each question pair as described by TELECI method:

N-P record when the user selects an incorrect answer before learning and the correct answer after learning;
P-P record when the user selects the correct answer before learning and the correct answer after learning;
N-N record when the user selects an incorrect answer before learning and an incorrect answer after learning;
P-N record when the user selects the correct answer before learning and an incorrect answer after learning.
Furthermore to create 3-dimensional space the N-N and P-N where summarized to create X-N dimension thus forming three main metrics (N-P, P-P, and X-N). Final dataframe is shown in table 2.

![image](https://user-images.githubusercontent.com/7984338/127029244-6e07f56f-1fd9-45fc-b115-9dfb1cd97427.png)
Table 2: Final dataframe of average probability for each unit.

This data frame is then used to form Telecides visualization using aforementioned Plotly library (figure 1).


![image](https://user-images.githubusercontent.com/7984338/127029293-92c591e3-58f3-4630-a9a4-b7e8d5f6a2f4.png)
Pic: 3D visualisation of Telecides.

To improve the convey of information the "complete learning acquisition landscape" had to be implemented. Based on scientific paper describing Telecides we created dataframe (Table 3) which consists of metrics used to describe this landscape.

![image](https://user-images.githubusercontent.com/7984338/127029325-92a35d45-0db9-4831-aae5-aeef28db4bed.png)
Table 3: Calculated theoretical values of average relative probability for N-P, P-P, X-N for three types of e-content (too complicated content, too easy content, and ideally matching course content).

By visualizing this dataframe we can aquire complete learning acquisition landscape in 3D space (Figure 2)

![image](https://user-images.githubusercontent.com/7984338/127029352-975b99ab-b3a9-494c-9292-08046b0fe0c3.png)
Pic: Complete learning acquisition landscape.

Final step was to introduce previously developed Telecides into the learning landscape (Figure 3).

![image](https://user-images.githubusercontent.com/7984338/127029390-c78d3754-9c89-4bc6-9c6f-4cde612ed224.png)
Pic: Final visualization of Telecides.



## Installation

Use the [git clone](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone) to install TELECI. Then use Python IDE of your choice to run localhost version of application.
Make sure all the necessary libraries from `requirements.txt` has been installed using [pip](https://pip.pypa.io/en/stable/).
```bash
git clone https://github.com/zanis-timsans/TELECI-APP
pip install library_name
```

## Usage
Use together with Moodle ARTSS xAPI plugin.

Update dropdown with your API link.
```python
dcc.Dropdown(options=[
                {'label': 'Your course',
                 'value': 'API link'}])
```

## Contributing
Pull requests are welcome when authorized. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Copyright (C) RTU Distance Education Study Center - All Rights Reserved\
Unauthorized copying of this file, via any medium is strictly prohibited\
Proprietary and confidential\
Part of the [TELECI project](https://teleci.lv/)\
Written by TELECI  Team 2019
