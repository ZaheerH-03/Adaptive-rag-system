from prompts.base import build_base_prompt

def few_shot_prompt(context: str, question: str) -> str:
    strategy = """
Examples:

Context:
Data integration is the process of combining data from different sources into a unified and consistent view. Organizations often store data in multiple systems such as databases, cloud applications, and files, which creates data silos. Data integration techniques bring this data together so it can be analyzed and used effectively.

A common approach to data integration is ETL (Extract, Transform, Load), where data is extracted from source systems, transformed into a standardized format, and loaded into a central repository such as a data warehouse. Modern platforms may also support real-time data integration and automated data pipelines.

Effective data integration improves data quality, enables comprehensive analytics, and supports business intelligence and decision-making across the organization.

Question:
What is data integration?

Answer:
Overview:
Data integration is the process of combining data from multiple sources into a unified and consistent view so it can be analyzed and used effectively. It helps organizations bring together data stored across different systems, reducing silos and enabling better access to information.

Details:
It typically involves methods such as ETL, where data is extracted from source systems, transformed into a standardized format, and loaded into a central repository like a data warehouse. Modern data integration platforms may also support real-time pipelines and automation. By integrating data, organizations improve data quality and enable analytics, business intelligence, and informed decision-making.

Context:
A Data Warehouse and a Data Lake serve distinct roles within a data ecosystem, primarily differing in their treatment of data structure and processing workflows. A Data Warehouse is a highly controlled environment designed for structured data that has been cleaned and transformed for a specific purpose, typically supporting business intelligence and standardized reporting. It operates on a Schema-on-Write basis, meaning the data must be formatted to fit a predefined structure before it can be stored. This makes it exceptionally fast for SQL-based queries and executive dashboards but less flexible for experimental analysis.

In contrast, a Data Lake is a vast, scalable repository that stores data in its natural, raw format—whether structured, semi-structured (like JSON or logs), or entirely unstructured (like images and PDFs). It utilizes a Schema-on-Read approach, where the structure is only applied when the data is pulled for use. This flexibility makes it the ideal landing zone for big data discovery and training machine learning models, as it preserves the original data fidelity for data scientists. While the warehouse offers a single version of the truth for business decisions, the lake provides a cost-effective, high-volume storage solution for deep exploration and advanced predictive analytics.

Question:
Differentiate between data lake and data warehouse.

Answer:
Overview:
A Data Warehouse and a Data Lake differ primarily in how they store, structure, and process data. A Data Warehouse stores structured, cleaned, and transformed data organized for reporting and business intelligence, while a Data Lake stores raw data in its original format, including structured, semi-structured, and unstructured data. The warehouse emphasizes consistency and fast querying for decision-making, whereas the lake prioritizes flexibility and large-scale storage for advanced analytics and data exploration.

Details:
A Data Warehouse operates on a schema-on-write approach, meaning data must be transformed and structured before storage, enabling efficient SQL queries and standardized reporting. In contrast, a Data Lake uses a schema-on-read approach, allowing raw data to be stored first and structured later when needed. Warehouses support business intelligence and executive dashboards by providing a consistent view of curated data, while data lakes serve as scalable repositories for big data, machine learning, and exploratory analysis. The warehouse ensures controlled, high-quality data for decisions, whereas the lake preserves full data fidelity for flexible analytical use.
"""
    return build_base_prompt(context, question, strategy)
