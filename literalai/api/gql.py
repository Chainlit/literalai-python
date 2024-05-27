from typing import List, Union

from literalai.helper import ensure_values_serializable
from literalai.step import Step, StepDict

STEP_FIELDS = """
        id
        threadId
        parentId
        startTime
        endTime
        createdAt
        type
        error
        input
        output
        metadata
        scores {
            id
            type
            name
            value
            comment
        }
        tags
        generation {
          prompt
          completion
          createdAt
          provider
          model
          variables
          messages
          messageCompletion
          tools
          settings
          stepId
          tokenCount              
          inputTokenCount         
          outputTokenCount        
          ttFirstToken          
          duration                
          tokenThroughputInSeconds
          error
          type
        }
        name
        attachments {
            id
            stepId
            metadata
            mime
            name
            objectKey
            url
        }"""

THREAD_FIELDS = (
    """
        id
        name
        metadata
        tags
        createdAt
        participant {
            id
            identifier
            metadata
        }
        steps {
"""
    + STEP_FIELDS
    + """
        }"""
)

SHALLOW_THREAD_FIELDS = """
        id
        name
        metadata
        tags
        createdAt
        participant {
            id
            identifier
            metadata
        }    
        
"""

GET_PARTICIPANTS = """query GetParticipants(
$after: ID,
$before: ID,
$cursorAnchor: DateTime,
$filters: [participantsInputType!],
$first: Int,
$last: Int,
$projectId: String,
) {
participants(
    after: $after,
    before: $before,
    cursorAnchor: $cursorAnchor,
    filters: $filters,
    first: $first,
    last: $last,
    projectId: $projectId,
    ) {
    pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
    }
    totalCount
    edges {
        cursor
        node {
            id
            createdAt
            lastEngaged
            threadCount
            tokenCount
            identifier
            metadata
        }
    }
}
}
"""

CREATE_PARTICIPANT = """mutation CreateUser($identifier: String!, $metadata: Json) {
createParticipant(identifier: $identifier, metadata: $metadata) {
    id
    identifier
    metadata
}
}
"""

UPDATE_PARTICIPANT = """mutation UpdateUser(
    $id: String!,
    $identifier: String,
    $metadata: Json,
) {
    updateParticipant(
        id: $id,
        identifier: $identifier,
        metadata: $metadata
    ) {
        id
        identifier
        metadata
    }
}
"""

GET_PARTICIPANT = """query GetUser($id: String, $identifier: String) {
participant(id: $id, identifier: $identifier) {
    id
    identifier
    metadata
    createdAt
}
}"""

DELETE_PARTICIPANT = """
mutation DeleteUser($id: String!) {
    deleteParticipant(id: $id) {
        id
    }
}
"""

GET_THREADS = (
    """
query GetThreads(
    $after: ID,
    $before: ID,
    $cursorAnchor: DateTime,
    $filters: [ThreadsInputType!],
    $orderBy: ThreadsOrderByInput,
    $first: Int,
    $last: Int,
    $projectId: String,
    $stepTypesToKeep: [StepType!],
    ) {
    threads(
        after: $after,
        before: $before,
        cursorAnchor: $cursorAnchor,
        filters: $filters,
        orderBy: $orderBy,
        first: $first,
        last: $last,
        projectId: $projectId,
        stepTypesToKeep: $stepTypesToKeep,
        ) {
        pageInfo {
            startCursor
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
        edges {
            cursor
            node {
"""
    + THREAD_FIELDS
    + """
            }
        }
    }
}
"""
)

LIST_THREADS = """query listThreads(
$first: Int
$after: ID
$last: Int
$before: ID
$skip: Int
$projectId: String
$filters: [ThreadsInputType!]
$orderBy: ThreadsOrderByInput
$cursorAnchor: DateTime
) {
threads(
    first: $first
    after: $after
    last: $last
    before: $before
    skip: $skip
    projectId: $projectId
    filters: $filters
    orderBy: $orderBy
    cursorAnchor: $cursorAnchor
) {
    pageInfo {
    hasNextPage
    hasPreviousPage
    startCursor
    endCursor
    }
    totalCount
    edges {
    node {
        id
        createdAt
        tokenCount
        name
        metadata
        duration
        tags
        participant {
        identifier
        id
        }
    }
    }
}
}"""

CREATE_THREAD = (
    """
mutation CreateThread(
    $name: String,
    $metadata: Json,
    $participantId: String,
    $environment: String,
    $tags: [String!],
) {
    createThread(
        name: $name
        metadata: $metadata
        participantId: $participantId
        environment: $environment
        tags: $tags
    ) {
"""
    + SHALLOW_THREAD_FIELDS
    + """
    }
}
"""
)

UPSERT_THREAD = (
    """
mutation UpsertThread(
    $id: String!,
    $name: String,
    $metadata: Json,
    $participantId: String,
    $environment: String,
    $tags: [String!],
) {
    upsertThread(
        id: $id
        name: $name
        metadata: $metadata
        participantId: $participantId
        environment: $environment
        tags: $tags
    ) {
"""
    + SHALLOW_THREAD_FIELDS
    + """
    }
}
"""
)

UPDATE_THREAD = (
    """
mutation UpdateThread(
    $id: String!,
    $name: String,
    $metadata: Json,
    $participantId: String,
    $environment: String,
    $tags: [String!],
) {
    updateThread(
        id: $id
        name: $name
        metadata: $metadata
        participantId: $participantId
        environment: $environment
        tags: $tags
    ) {
"""
    + SHALLOW_THREAD_FIELDS
    + """
    }
}
"""
)

GET_THREAD = (
    """
query GetThread($id: String!) {
    threadDetail(id: $id) {
"""
    + THREAD_FIELDS
    + """
    }
}
"""
)

DELETE_THREAD = """
mutation DeleteThread($thread_id: String!) {
    deleteThread(id: $thread_id) {
        id
    }
}
"""

GET_SCORES = """
query GetScores(
    $after: ID,
    $before: ID,
    $cursorAnchor: DateTime,
    $filters: [scoresInputType!],
    $orderBy: ScoresOrderByInput,
    $first: Int,
    $last: Int,
    $projectId: String,
    ) {
    scores(
        after: $after,
        before: $before,
        cursorAnchor: $cursorAnchor,
        filters: $filters,
        orderBy: $orderBy,
        first: $first,
        last: $last,
        projectId: $projectId,
        ) {
        pageInfo {
            startCursor
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
        edges {
            cursor
            node {
                comment
                createdAt
                id
                projectId
                stepId
                generationId
                datasetExperimentItemId
                type
                updatedAt
                name
                value
                tags
                step {
                    thread {
                    id
                    participant {
                        identifier
                            }
                        }
                    }
                }
            }
        }
    }
"""

CREATE_SCORE = """
mutation CreateScore(
    $name: String!,
    $type: ScoreType!,
    $value: Float!,
    $stepId: String,
    $generationId: String,
    $datasetExperimentItemId: String,
    $comment: String,
    $tags: [String!],

) {
    createScore(
        name: $name,
        type: $type,
        value: $value,
        stepId: $stepId,
        generationId: $generationId,
        datasetExperimentItemId: $datasetExperimentItemId,
        comment: $comment,
        tags: $tags,
    ) {
        id
        name,
        type,
        value,
        stepId,
        generationId,
        datasetExperimentItemId,
        comment,
        tags,
    }
}
"""

UPDATE_SCORE = """
mutation UpdateScore(
    $id: String!,
    $comment: String,
    $value: Float!,
) {
    updateScore(
        id: $id,
        comment: $comment,
        value: $value,
    ) {
        id
        name,
        type,
        value,
        stepId,
        generationId,
        datasetExperimentItemId,
        comment
    }
}
"""

DELETE_SCORE = """
mutation DeleteScore($id: String!) {
    deleteScore(id: $id) {
        id
    }
}
"""

CREATE_ATTACHMENT = """
mutation CreateAttachment(
    $metadata: Json,
    $mime: String,
    $name: String!,
    $objectKey: String,
    $stepId: String!,
    $url: String,
) {
    createAttachment(
        metadata: $metadata,
        mime: $mime,
        name: $name,
        objectKey: $objectKey,
        stepId: $stepId,
        url: $url,
    ) {
        id
        threadId
        stepId
        metadata
        mime
        name
        objectKey
        url
    }
}
"""

UPDATE_ATTACHMENT = """
mutation UpdateAttachment(
    $id: String!,
    $metadata: Json,
    $mime: String,
    $name: String,
    $objectKey: String,
    $projectId: String,
    $url: String,
) {
    updateAttachment(
        id: $id,
        metadata: $metadata,
        mime: $mime,
        name: $name,
        objectKey: $objectKey,
        projectId: $projectId,
        url: $url,
    ) {
        id
        threadId
        stepId
        metadata
        mime
        name
        objectKey
        url
    }
}
"""

GET_ATTACHMENT = """
query GetAttachment($id: String!) {
    attachment(id: $id) {
        id
        threadId
        stepId
        metadata
        mime
        name
        objectKey
        url
    }
}
"""

DELETE_ATTACHMENT = """
mutation DeleteAttachment($id: String!) {
    deleteAttachment(id: $id) {
        id
    }
}
"""

CREATE_STEP = (
    """
mutation CreateStep(
    $threadId: String,
    $type: StepType,
    $startTime: DateTime,
    $endTime: DateTime,
    $input: Json,
    $output: Json,
    $metadata: Json,
    $parentId: String,
    $name: String,
    $tags: [String!],

) {
    createStep(
        threadId: $threadId,
        type: $type,
        startTime: $startTime,
        endTime: $endTime,
        input: $input,
        output: $output,
        metadata: $metadata,
        parentId: $parentId,
        name: $name,
        tags: $tags,
    ) {
"""
    + STEP_FIELDS
    + """
    }
}
"""
)

UPDATE_STEP = (
    """
mutation UpdateStep(
    $id: String!,
    $type: StepType,
    $input: Json,
    $output: Json,
    $metadata: Json,
    $name: String,
    $startTime: DateTime,
    $endTime: DateTime,
    $parentId: String,
    $tags: [String!],
) {
    updateStep(
        id: $id,
        type: $type,
        startTime: $startTime,
        endTime: $endTime,
        input: $input,
        output: $output,
        metadata: $metadata,
        name: $name,
        tags: $tags,
        parentId: $parentId,
    ) {    
"""
    + STEP_FIELDS
    + """
    }
}
"""
)

GET_STEPS = (
    """
query GetSteps(
    $after: ID,
    $before: ID,
    $cursorAnchor: DateTime,
    $filters: [stepsInputType!],
    $orderBy: StepsOrderByInput,
    $first: Int,
    $last: Int,
    $projectId: String,
    ) {
    steps(
        after: $after,
        before: $before,
        cursorAnchor: $cursorAnchor,
        filters: $filters,
        orderBy: $orderBy,
        first: $first,
        last: $last,
        projectId: $projectId,
        ) {
        pageInfo {
            startCursor
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
        edges {
            cursor
            node {
"""
    + STEP_FIELDS
    + """
            }
        }
    }
}
"""
)

GET_STEP = (
    """
query GetStep($id: String!) {
    step(id: $id) {"""
    + STEP_FIELDS
    + """
    }
}
"""
)

DELETE_STEP = """
mutation DeleteStep($id: String!) {
    deleteStep(id: $id) {
        id
    }
}
"""

GET_GENERATIONS = """
query GetGenerations(
    $after: ID,
    $before: ID,
    $cursorAnchor: DateTime,
    $filters: [generationsInputType!],
    $orderBy: GenerationsOrderByInput,
    $first: Int,
    $last: Int,
    $projectId: String,
    ) {
    generations(
        after: $after,
        before: $before,
        cursorAnchor: $cursorAnchor,
        filters: $filters,
        orderBy: $orderBy,
        first: $first,
        last: $last,
        projectId: $projectId,
        ) {
        pageInfo {
            startCursor
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
        edges {
            cursor
            node {
                id
                projectId
                prompt
                completion
                createdAt
                provider
                model
                variables
                messages
                messageCompletion
                tools
                settings
                stepId
                tokenCount
                duration
                inputTokenCount
                outputTokenCount
                ttFirstToken
                duration
                tokenThroughputInSeconds
                error
                type
                tags
                step {
                    threadId
                    thread {
                    participant {
                        identifier
                            }
                        }
                    }
                }
            }
        }
    }
"""

CREATE_GENERATION = """
mutation CreateGeneration($generation: GenerationPayloadInput!) {
    createGeneration(generation: $generation) {
        id,
        type
    }
}
"""

CREATE_DATASET = """
mutation createDataset(
    $name: String!
    $description: String
    $metadata: Json
    $type: DatasetType
) {
    createDataset(
        name: $name
        description: $description
        metadata: $metadata
        type: $type
    ) {
        id
        createdAt
        name
        description
        metadata
        type
    }
}
"""

UPDATE_DATASET = """
mutation UpdateDataset(
    $id: String!
    $name: String
    $description: String
    $metadata: Json
) {
    updateDataset(
        id: $id
        name: $name
        description: $description
        metadata: $metadata
    ) {
        id
        createdAt
        name
        description
        metadata
        type
    }
}
"""

DELETE_DATASET = """
    mutation DeleteDataset(
        $id: String!
    ) {
        deleteDataset(
            id: $id
        ) {
            id
            createdAt
            name
            description
            metadata
            type
        }
    }
"""

CREATE_EXPERIMENT = """
    mutation CreateDatasetExperiment(
        $name: String! 
        $datasetId: String!
        $promptId: String
        $params: Json
    ) {
        createDatasetExperiment(
            name: $name
            datasetId: $datasetId
            promptId: $promptId
            params: $params
        ) {
          id
          name
          datasetId
          params
        }
    }
"""

CREATE_EXPERIMENT_ITEM = """
    mutation CreateDatasetExperimentItem(
        $datasetExperimentId: String!
        $datasetItemId: String!
        $input: Json
        $output: Json
    ) {
        createDatasetExperimentItem(
            datasetExperimentId: $datasetExperimentId
            datasetItemId: $datasetItemId
            input: $input
            output: $output
        ) {
          id
          input
          output
        }
      }
"""

CREATE_DATASET_ITEM = """
mutation CreateDatasetItem(
    $datasetId: String!
    $input: Json!
    $expectedOutput: Json
    $metadata: Json
) {
    createDatasetItem(
        datasetId: $datasetId
        input: $input
        expectedOutput: $expectedOutput
        metadata: $metadata
    ) {
        id
        createdAt
        datasetId
        metadata
        input
        expectedOutput
        intermediarySteps
    }
}
"""

GET_DATASET_ITEM = """
query GetDataItem($id: String!) {
    datasetItem(id: $id) {
        id
        createdAt
        datasetId
        metadata
        input
        expectedOutput
        intermediarySteps
    }
}
"""

DELETE_DATASET_ITEM = """
mutation DeleteDatasetItem($id: String!) {
    deleteDatasetItem(id: $id) {
        id
        createdAt
        datasetId
        metadata
        input
        expectedOutput
        intermediarySteps
    }
}
"""

ADD_STEP_TO_DATASET = """
mutation AddStepToDataset(
    $datasetId: String!
    $stepId: String!
    $metadata: Json
) {
    addStepToDataset(
        datasetId: $datasetId
        stepId: $stepId
        metadata: $metadata
    ) {
        id
        createdAt
        datasetId
        metadata
        input
        expectedOutput
        intermediarySteps
    }
}
"""

ADD_GENERATION_TO_DATASET = """
mutation AddGenerationToDataset(
    $datasetId: String!
    $generationId: String!
    $metadata: Json
) {
    addGenerationToDataset(
        datasetId: $datasetId
        generationId: $generationId
        metadata: $metadata
    ) {
        id
        createdAt
        datasetId
        metadata
        input
        expectedOutput
        intermediarySteps
    }
}
"""

CREATE_PROMPT_LINEAGE = """mutation createPromptLineage(
    $name: String!
    $description: String
  ) {
    createPromptLineage(
      name: $name
      description: $description
    ) {
      id
      name
    }
  }"""

CREATE_PROMPT_VERSION = """mutation createPromptVersion(
    $lineageId: String!
    $versionDesc: String
    $templateMessages: Json
    $tools: Json
    $settings: Json
    $variables: Json
    $variablesDefaultValues: Json
  ) {
    createPromptVersion(
      lineageId: $lineageId
      versionDesc: $versionDesc
      templateMessages: $templateMessages
      tools: $tools
      settings: $settings
      variables: $variables
      variablesDefaultValues: $variablesDefaultValues
    ) {
      id
      lineage {
        name
      }
      version
      createdAt
      tools
      settings
      templateMessages
    }
  }"""

GET_PROMPT_VERSION = """
query GetPrompt($id: String, $name: String, $version: Int) {
    promptVersion(id: $id, name: $name, version: $version) {
        id
        createdAt
        updatedAt
        type
        templateMessages
        tools
        settings
        variables
        variablesDefaultValues
        version
        lineage {
            name
        }
    }
}
"""

PROMOTE_PROMPT_VERSION = """mutation promotePromptVersion(
    $lineageId: String!
    $version: Int!
  ) {
    promotePromptVersion(
      lineageId: $lineageId
      version: $version
    ) {
      id
      championId
    }
  }"""


def serialize_step(event, id):
    result = {}

    for key, value in event.items():
        # Only keep the keys that are not None to avoid overriding existing values
        if value is not None:
            result[f"{key}_{id}"] = value

    return result


def steps_variables_builder(steps: List[Union["StepDict", "Step"]]):
    variables = {}
    for i in range(len(steps)):
        step = steps[i]
        if isinstance(step, Step):
            if step.input:
                step.input = ensure_values_serializable(step.input)
            if step.output:
                step.output = ensure_values_serializable(step.output)
            variables.update(serialize_step(step.to_dict(), i))
        else:
            if step.get("input"):
                step["input"] = ensure_values_serializable(step["input"])
            if step.get("output"):
                step["output"] = ensure_values_serializable(step["output"])
            variables.update(serialize_step(step, i))
    return variables


def steps_query_variables_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""$id_{id}: String!
        $threadId_{id}: String
        $type_{id}: StepType
        $startTime_{id}: DateTime
        $endTime_{id}: DateTime
        $error_{id}: String
        $input_{id}: Json
        $output_{id}: Json
        $metadata_{id}: Json
        $parentId_{id}: String
        $name_{id}: String
        $tags_{id}: [String!]
        $generation_{id}: GenerationPayloadInput
        $scores_{id}: [ScorePayloadInput!]
        $attachments_{id}: [AttachmentPayloadInput!]
        """
    return generated


def steps_ingest_steps_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""
      step{id}: ingestStep(
        id: $id_{id}
        threadId: $threadId_{id}
        startTime: $startTime_{id}
        endTime: $endTime_{id}
        type: $type_{id}
        error: $error_{id}
        input: $input_{id}
        output: $output_{id}
        metadata: $metadata_{id}
        parentId: $parentId_{id}
        name: $name_{id}
        tags: $tags_{id}
        generation: $generation_{id}
        scores: $scores_{id}
        attachments: $attachments_{id}
      ) {{
        ok
        message
      }}
"""
    return generated


def steps_query_builder(steps):
    return f"""
    mutation AddStep({steps_query_variables_builder(steps)}) {{
      {steps_ingest_steps_builder(steps)}
    }}
    """
